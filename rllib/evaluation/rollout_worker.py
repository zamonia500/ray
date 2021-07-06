import random
import numpy as np
import gym
import logging
import pickle
import platform
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar, \
    TYPE_CHECKING, Union

import ray
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.env.env_context import EnvContext
from ray.rllib.env.external_env import ExternalEnv
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.env.external_multi_agent_env import ExternalMultiAgentEnv
from ray.rllib.env.utils import record_env_wrapper
from ray.rllib.env.vector_env import VectorEnv
from ray.rllib.env.wrappers.atari_wrappers import wrap_deepmind, is_atari
from ray.rllib.evaluation.sampler import AsyncSampler, SyncSampler
from ray.rllib.evaluation.rollout_metrics import RolloutMetrics
from ray.rllib.models import ModelCatalog
from ray.rllib.models.preprocessors import NoPreprocessor, Preprocessor
from ray.rllib.offline import NoopOutput, IOContext, OutputWriter, InputReader
from ray.rllib.offline.off_policy_estimator import OffPolicyEstimator, \
    OffPolicyEstimate
from ray.rllib.offline.is_estimator import ImportanceSamplingEstimator
from ray.rllib.offline.wis_estimator import WeightedImportanceSamplingEstimator
from ray.rllib.policy.sample_batch import MultiAgentBatch, DEFAULT_POLICY_ID
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.tf_policy import TFPolicy
from ray.rllib.policy.torch_policy import TorchPolicy
from ray.rllib.utils import merge_dicts
from ray.rllib.utils.annotations import DeveloperAPI
from ray.rllib.utils.debug import summarize
from ray.rllib.utils.deprecation import DEPRECATED_VALUE, deprecation_warning
from ray.rllib.utils.filter import get_filter, Filter
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.sgd import do_minibatch_sgd
from ray.rllib.utils.tf_run_builder import TFRunBuilder
from ray.rllib.utils.typing import AgentID, EnvConfigDict, EnvType, \
    ModelConfigDict, ModelGradients, ModelWeights, \
    MultiAgentPolicyConfigDict, PartialTrainerConfigDict, PolicyID, \
    SampleBatchType, TrainerConfigDict
from ray.util.debug import log_once, disable_log_once_globally, \
    enable_periodic_logging
from ray.util.iter import ParallelIteratorWorker

if TYPE_CHECKING:
    from ray.rllib.evaluation.episode import MultiAgentEpisode
    from ray.rllib.evaluation.observation_function import ObservationFunction
    from ray.rllib.agents.callbacks import DefaultCallbacks  # noqa

# Generic type var for foreach_* methods.
T = TypeVar("T")

tf1, tf, tfv = try_import_tf()
torch, _ = try_import_torch()

logger = logging.getLogger(__name__)

# Handle to the current rollout worker, which will be set to the most recently
# created RolloutWorker in this process. This can be helpful to access in
# custom env or policy classes for debugging or advanced use cases.
_global_worker: "RolloutWorker" = None


@DeveloperAPI
def get_global_worker() -> "RolloutWorker":
    """Returns a handle to the active rollout worker in this process."""

    global _global_worker
    return _global_worker


@DeveloperAPI
class RolloutWorker(ParallelIteratorWorker):
    """Common experience collection class.

    This class wraps a policy instance and an environment class to
    collect experiences from the environment. You can create many replicas of
    this class as Ray actors to scale RL training.

    This class supports vectorized and multi-agent policy evaluation (e.g.,
    VectorEnv, MultiAgentEnv, etc.)

    Examples:
        >>> # Create a rollout worker and using it to collect experiences.
        >>> worker = RolloutWorker(
        ...   env_creator=lambda _: gym.make("CartPole-v0"),
        ...   policy_spec=PGTFPolicy)
        >>> print(worker.sample())
        SampleBatch({
            "obs": [[...]], "actions": [[...]], "rewards": [[...]],
            "dones": [[...]], "new_obs": [[...]]})

        >>> # Creating a multi-agent rollout worker
        >>> worker = RolloutWorker(
        ...   env_creator=lambda _: MultiAgentTrafficGrid(num_cars=25),
        ...   policy_spec={
        ...       # Use an ensemble of two policies for car agents
        ...       "car_policy1":
        ...         (PGTFPolicy, Box(...), Discrete(...), {"gamma": 0.99}),
        ...       "car_policy2":
        ...         (PGTFPolicy, Box(...), Discrete(...), {"gamma": 0.95}),
        ...       # Use a single shared policy for all traffic lights
        ...       "traffic_light_policy":
        ...         (PGTFPolicy, Box(...), Discrete(...), {}),
        ...   },
        ...   policy_mapping_fn=lambda agent_id, episode, **kwargs:
        ...     random.choice(["car_policy1", "car_policy2"])
        ...     if agent_id.startswith("car_") else "traffic_light_policy")
        >>> print(worker.sample())
        MultiAgentBatch({
            "car_policy1": SampleBatch(...),
            "car_policy2": SampleBatch(...),
            "traffic_light_policy": SampleBatch(...)})
    """

    @DeveloperAPI
    @classmethod
    def as_remote(cls,
                  num_cpus: int = None,
                  num_gpus: int = None,
                  memory: int = None,
                  object_store_memory: int = None,
                  resources: dict = None) -> type:
        return ray.remote(
            num_cpus=num_cpus,
            num_gpus=num_gpus,
            memory=memory,
            object_store_memory=object_store_memory,
            resources=resources)(cls)

    @DeveloperAPI
    def __init__(
            self,
            *,
            env_creator: Callable[[EnvContext], EnvType],
            validate_env: Optional[Callable[[EnvType, EnvContext],
                                            None]] = None,
            policy_spec: Union[type, Dict[
                str, Tuple[Optional[type], gym.Space, gym.Space,
                           PartialTrainerConfigDict]]] = None,
            policy_mapping_fn: Optional[Callable[
                [AgentID, "MultiAgentEpisode"], PolicyID]] = None,
            policies_to_train: Optional[List[PolicyID]] = None,
            tf_session_creator: Optional[Callable[[], "tf1.Session"]] = None,
            rollout_fragment_length: int = 100,
            count_steps_by: str = "env_steps",
            batch_mode: str = "truncate_episodes",
            episode_horizon: int = None,
            preprocessor_pref: str = "deepmind",
            sample_async: bool = False,
            compress_observations: bool = False,
            num_envs: int = 1,
            observation_fn: "ObservationFunction" = None,
            observation_filter: str = "NoFilter",
            clip_rewards: bool = None,
            normalize_actions: bool = True,
            clip_actions: bool = False,
            env_config: EnvConfigDict = None,
            model_config: ModelConfigDict = None,
            policy_config: TrainerConfigDict = None,
            worker_index: int = 0,
            num_workers: int = 0,
            record_env: Union[bool, str] = False,
            log_dir: str = None,
            log_level: str = None,
            callbacks: Type["DefaultCallbacks"] = None,
            input_creator: Callable[[
                IOContext
            ], InputReader] = lambda ioctx: ioctx.default_sampler_input(),
            input_evaluation: List[str] = frozenset([]),
            output_creator: Callable[
                [IOContext], OutputWriter] = lambda ioctx: NoopOutput(),
            remote_worker_envs: bool = False,
            remote_env_batch_wait_ms: int = 0,
            soft_horizon: bool = False,
            no_done_at_end: bool = False,
            seed: int = None,
            extra_python_environs: dict = None,
            fake_sampler: bool = False,
            spaces: Optional[Dict[PolicyID, Tuple[gym.spaces.Space,
                                                  gym.spaces.Space]]] = None,
            policy: Union[type, Dict[
                str, Tuple[Optional[type], gym.Space, gym.Space,
                           PartialTrainerConfigDict]]] = None,
            monitor_path=None,
    ):
        """Initialize a rollout worker.

        Args:
            env_creator (Callable[[EnvContext], EnvType]): Function that
                returns a gym.Env given an EnvContext wrapped configuration.
            validate_env (Optional[Callable[[EnvType, EnvContext], None]]):
                Optional callable to validate the generated environment (only
                on worker=0).
            policy_spec (Union[type, Dict[str, Tuple[Type[Policy], gym.Space,
                gym.Space, PartialTrainerConfigDict]]]): Either a Policy class
                or a dict of policy id strings to
                (Policy class, obs_space, action_space, config)-tuples. If a
                dict is specified, then we are in multi-agent mode and a
                policy_mapping_fn can also be set (if not, will map all agents
                to DEFAULT_POLICY_ID).
            policy_mapping_fn (Optional[Callable[[AgentID, MultiAgentEpisode],
                PolicyID]]): A callable that maps agent ids to policy ids in
                multi-agent mode. This function will be called each time a new
                agent appears in an episode, to bind that agent to a policy
                for the duration of the episode. If not provided, will map all
                agents to DEFAULT_POLICY_ID.
            policies_to_train (Optional[List[PolicyID]]): Optional list of
                policies to train, or None for all policies.
            tf_session_creator (Optional[Callable[[], tf1.Session]]): A
                function that returns a TF session. This is optional and only
                useful with TFPolicy.
            rollout_fragment_length (int): The target number of steps
                (maesured in `count_steps_by`) to include in each sample
                batch returned from this worker.
            count_steps_by (str): The unit in which to count fragment
                lengths. One of env_steps or agent_steps.
            batch_mode (str): One of the following batch modes:
                "truncate_episodes": Each call to sample() will return a batch
                    of at most `rollout_fragment_length * num_envs` in size.
                    The batch will be exactly
                    `rollout_fragment_length * num_envs` in size if
                    postprocessing does not change batch sizes. Episodes may be
                    truncated in order to meet this size requirement.
                "complete_episodes": Each call to sample() will return a batch
                    of at least `rollout_fragment_length * num_envs` in size.
                    Episodes will not be truncated, but multiple episodes may
                    be packed within one batch to meet the batch size. Note
                    that when `num_envs > 1`, episode steps will be buffered
                    until the episode completes, and hence batches may contain
                    significant amounts of off-policy data.
            episode_horizon (int): Whether to stop episodes at this horizon.
            preprocessor_pref (str): Whether to prefer RLlib preprocessors
                ("rllib") or deepmind ("deepmind") when applicable.
            sample_async (bool): Whether to compute samples asynchronously in
                the background, which improves throughput but can cause samples
                to be slightly off-policy.
            compress_observations (bool): If true, compress the observations.
                They can be decompressed with rllib/utils/compression.
            num_envs (int): If more than one, will create multiple envs
                and vectorize the computation of actions. This has no effect if
                if the env already implements VectorEnv.
            observation_fn (ObservationFunction): Optional multi-agent
                observation function.
            observation_filter (str): Name of observation filter to use.
            clip_rewards (bool): Whether to clip rewards to [-1, 1] prior to
                experience postprocessing. Setting to None means clip for Atari
                only.
            normalize_actions (bool): Whether to normalize actions to the
                action space's bounds.
            clip_actions (bool): Whether to clip action values to the range
                specified by the policy action space.
            env_config (EnvConfigDict): Config to pass to the env creator.
            model_config (ModelConfigDict): Config to use when creating the
                policy model.
            policy_config (TrainerConfigDict): Config to pass to the policy.
                In the multi-agent case, this config will be merged with the
                per-policy configs specified by `policy_spec`.
            worker_index (int): For remote workers, this should be set to a
                non-zero and unique value. This index is passed to created envs
                through EnvContext so that envs can be configured per worker.
            num_workers (int): For remote workers, how many workers altogether
                have been created?
            record_env (Union[bool, str]): Write out episode stats and videos
                using gym.wrappers.Monitor to this directory if specified. If
                True, use the default output dir in ~/ray_results/.... If
                False, do not record anything.
            log_dir (str): Directory where logs can be placed.
            log_level (str): Set the root log level on creation.
            callbacks (Type[DefaultCallbacks]): Custom sub-class of
                DefaultCallbacks for training/policy/rollout-worker callbacks.
            input_creator (Callable[[IOContext], InputReader]): Function that
                returns an InputReader object for loading previous generated
                experiences.
            input_evaluation (List[str]): How to evaluate the policy
                performance. This only makes sense to set when the input is
                reading offline data. The possible values include:
                  - "is": the step-wise importance sampling estimator.
                  - "wis": the weighted step-wise is estimator.
                  - "simulation": run the environment in the background, but
                    use this data for evaluation only and never for learning.
            output_creator (Callable[[IOContext], OutputWriter]): Function that
                returns an OutputWriter object for saving generated
                experiences.
            remote_worker_envs (bool): If using num_envs_per_worker > 1,
                whether to create those new envs in remote processes instead of
                in the current process. This adds overheads, but can make sense
                if your envs are expensive to step/reset (e.g., for StarCraft).
                Use this cautiously, overheads are significant!
            remote_env_batch_wait_ms (float): Timeout that remote workers
                are waiting when polling environments. 0 (continue when at
                least one env is ready) is a reasonable default, but optimal
                value could be obtained by measuring your environment
                step / reset and model inference perf.
            soft_horizon (bool): Calculate rewards but don't reset the
                environment when the horizon is hit.
            no_done_at_end (bool): Ignore the done=True at the end of the
                episode and instead record done=False.
            seed (int): Set the seed of both np and tf to this value to
                to ensure each remote worker has unique exploration behavior.
            extra_python_environs (dict): Extra python environments need to
                be set.
            fake_sampler (bool): Use a fake (inf speed) sampler for testing.
            spaces (Optional[Dict[PolicyID, Tuple[gym.spaces.Space,
                gym.spaces.Space]]]): An optional space dict mapping policy IDs
                to (obs_space, action_space)-tuples. This is used in case no
                Env is created on this RolloutWorker.
            policy: Obsoleted arg. Use `policy_spec` instead.
            monitor_path: Obsoleted arg. Use `record_env` instead.
        """
        # Deprecated args.
        if policy is not None:
            deprecation_warning("policy", "policy_spec", error=False)
            policy_spec = policy
        assert policy_spec is not None, "Must provide `policy_spec` when " \
                                        "creating RolloutWorker!"
        if monitor_path is not None:
            deprecation_warning("monitor_path", "record_env", error=False)
            record_env = monitor_path

        self._original_kwargs: dict = locals().copy()
        del self._original_kwargs["self"]

        global _global_worker
        _global_worker = self

        # set extra environs first
        if extra_python_environs:
            for key, value in extra_python_environs.items():
                os.environ[key] = str(value)

        def gen_rollouts():
            while True:
                yield self.sample()

        ParallelIteratorWorker.__init__(self, gen_rollouts, False)

        policy_config: TrainerConfigDict = policy_config or {}
        if (tf1 and policy_config.get("framework") in ["tf2", "tfe"]
                # This eager check is necessary for certain all-framework tests
                # that use tf's eager_mode() context generator.
                and not tf1.executing_eagerly()):
            tf1.enable_eager_execution()

        if log_level:
            logging.getLogger("ray.rllib").setLevel(log_level)

        if worker_index > 1:
            disable_log_once_globally()  # only need 1 worker to log
        elif log_level == "DEBUG":
            enable_periodic_logging()

        env_context = EnvContext(
            env_config or {}, worker_index, num_workers=num_workers)
        self.env_context = env_context
        self.policy_config: TrainerConfigDict = policy_config
        if callbacks:
            self.callbacks: "DefaultCallbacks" = callbacks()
        else:
            from ray.rllib.agents.callbacks import DefaultCallbacks  # noqa
            self.callbacks: DefaultCallbacks = DefaultCallbacks()
        self.worker_index: int = worker_index
        self.num_workers: int = num_workers
        model_config: ModelConfigDict = model_config or {}

        # Default policy mapping fn is to always return DEFAULT_POLICY_ID,
        # independent on the agent ID and the episode passed in.
        self.policy_mapping_fn = lambda aid, ep, **kwargs: DEFAULT_POLICY_ID
        self.set_policy_mapping_fn(policy_mapping_fn)

        self.env_creator: Callable[[EnvContext], EnvType] = env_creator
        self.rollout_fragment_length: int = rollout_fragment_length * num_envs
        self.count_steps_by: str = count_steps_by
        self.batch_mode: str = batch_mode
        self.compress_observations: bool = compress_observations
        self.preprocessing_enabled: bool = True
        self.observation_filter = observation_filter
        self.last_batch: SampleBatchType = None
        self.global_vars: dict = None
        self.fake_sampler: bool = fake_sampler

        self.env = None

        # Create an env for this worker.
        if not (worker_index == 0 and num_workers > 0
                and policy_config["create_env_on_driver"] is False):
            # Run the `env_creator` function passing the EnvContext.
            self.env = env_creator(env_context)
        if self.env is not None:
            # Validate environment (general validation function).
            self.env = _validate_env(self.env)
            # Custom validation function given.
            if validate_env is not None:
                validate_env(self.env, self.env_context)

            # MultiAgentEnv (a gym.Env) -> Wrap and make
            # the wrapped Env yet another MultiAgentEnv.
            if isinstance(self.env, MultiAgentEnv):

                def wrap(env):
                    cls = env.__class__
                    # Add gym.Env as mixin parent to the env's class
                    # (so it can be wrapped).
                    env.__class__ = \
                        type(env.__class__.__name__, (type(env), gym.Env), {})
                    # Wrap the (now gym.Env) env with our (multi-agent capable)
                    # recording wrapper.
                    env = record_env_wrapper(env, record_env, log_dir,
                                             policy_config)
                    # Make sure, we make the wrapped object a member of the
                    # original MultiAgentEnv sub-class again.
                    if type(env) is not cls:
                        env.__class__ = \
                            type(cls.__name__, (type(env), cls), {})
                    return env

            # We can't auto-wrap a BaseEnv.
            elif isinstance(self.env, BaseEnv):

                def wrap(env):
                    return env

            # Atari type env and "deepmind" preprocessor pref.
            elif is_atari(self.env) and \
                    not model_config.get("custom_preprocessor") and \
                    preprocessor_pref == "deepmind":

                # Deepmind wrappers already handle all preprocessing.
                self.preprocessing_enabled = False

                # If clip_rewards not explicitly set to False, switch it
                # on here (clip between -1.0 and 1.0).
                if clip_rewards is None:
                    clip_rewards = True

                # Deprecated way of framestacking is used.
                framestack = model_config.get("framestack") is True
                # framestacking via trajectory view API is enabled.
                num_framestacks = model_config.get("num_framestacks", 0)

                # Trajectory view API is on and num_framestacks=auto:
                # Only stack traj. view based if old
                # `framestack=[invalid value]`.
                if num_framestacks == "auto":
                    if framestack == DEPRECATED_VALUE:
                        model_config["num_framestacks"] = num_framestacks = 4
                    else:
                        model_config["num_framestacks"] = num_framestacks = 0
                framestack_traj_view = num_framestacks > 1

                def wrap(env):
                    env = wrap_deepmind(
                        env,
                        dim=model_config.get("dim"),
                        framestack=framestack,
                        framestack_via_traj_view_api=framestack_traj_view)
                    env = record_env_wrapper(env, record_env, log_dir,
                                             policy_config)
                    return env

            # gym.Env -> Wrap with gym Monitor.
            else:

                def wrap(env):
                    return record_env_wrapper(env, record_env, log_dir,
                                              policy_config)

            # Wrap env through the correct wrapper.
            self.env: EnvType = wrap(self.env)

        def make_env(vector_index):
            return wrap(
                env_creator(
                    env_context.copy_with_overrides(
                        worker_index=worker_index,
                        vector_index=vector_index,
                        remote=remote_worker_envs)))

        self.make_env_fn = make_env

        self.tf_sess = None
        policy_dict = _validate_and_canonicalize(
            policy_spec, self.env, spaces=spaces, policy_config=policy_config)
        # List of IDs of those policies, which should be trained.
        # By default, these are all policies found in the policy_dict.
        self.policies_to_train: List[PolicyID] = policies_to_train or list(
            policy_dict.keys())
        self.set_policies_to_train(self.policies_to_train)

        self.policy_map: Dict[PolicyID, Policy] = None
        self.preprocessors: Dict[PolicyID, Preprocessor] = None

        # Set Python random, numpy, env, and torch/tf seeds.
        if seed is not None:
            # Python random module.
            random.seed(seed)
            # Numpy.
            np.random.seed(seed)
            # Gym.env.
            # This will silently fail for most OpenAI gyms
            # (they do nothing and return None per default)
            if not hasattr(self.env, "seed"):
                logger.info("Env doesn't support env.seed(): {}".format(
                    self.env))
            else:
                self.env.seed(seed)

            # Torch.
            if torch and policy_config.get("framework") == "torch":
                torch.manual_seed(seed)
                # See https://github.com/pytorch/pytorch/issues/47672.
                cuda_version = torch.version.cuda
                if cuda_version is not None and float(
                        torch.version.cuda) >= 10.2:
                    os.environ["CUBLAS_WORKSPACE_CONFIG"] = "4096:8"
                else:
                    from distutils.version import LooseVersion
                    if LooseVersion(
                            torch.__version__) >= LooseVersion("1.8.0"):
                        # Not all Operations support this.
                        torch.use_deterministic_algorithms(True)
                    else:
                        torch.set_determinstic(True)
                # This is only for Convolution no problem.
                torch.backends.cudnn.deterministic = True
            # Tf2.x.
            elif tf and policy_config.get("framework") == "tf2":
                tf.random.set_seed(seed)
            # Tf-eager.
            elif tf1 and policy_config.get("framework") == "tfe":
                tf1.set_random_seed(seed)

        if _has_tensorflow_graph(policy_dict) and not (
                tf1 and tf1.executing_eagerly()):
            if not tf1:
                raise ImportError("Could not import tensorflow")
            with tf1.Graph().as_default():
                if tf_session_creator:
                    self.tf_sess = tf_session_creator()
                else:
                    self.tf_sess = tf1.Session(
                        config=tf1.ConfigProto(
                            gpu_options=tf1.GPUOptions(allow_growth=True)))
                with self.tf_sess.as_default():
                    # set graph-level seed
                    if seed is not None:
                        tf1.set_random_seed(seed)
                    self.policy_map, self.preprocessors = \
                        self._build_policy_map(policy_dict, policy_config)
        else:
            self.policy_map, self.preprocessors = self._build_policy_map(
                policy_dict, policy_config)

        # Update Policy's view requirements from Model, only if Policy directly
        # inherited from base `Policy` class. At this point here, the Policy
        # must have it's Model (if any) defined and ready to output an initial
        # state.
        for pol in self.policy_map.values():
            if not pol._model_init_state_automatically_added:
                pol._update_model_view_requirements_from_init_state()

        if (ray.is_initialized()
                and ray.worker._mode() != ray.worker.LOCAL_MODE):
            # Check available number of GPUs
            if not ray.get_gpu_ids():
                logger.debug("Creating policy evaluation worker {}".format(
                    worker_index) +
                             " on CPU (please ignore any CUDA init errors)")
            elif (policy_config["framework"] in ["tf2", "tf", "tfe"] and
                  not tf.config.experimental.list_physical_devices("GPU")) or \
                    (policy_config["framework"] == "torch" and
                     not torch.cuda.is_available()):
                raise RuntimeError(
                    "GPUs were assigned to this worker by Ray, but "
                    "your DL framework ({}) reports GPU acceleration is "
                    "disabled. This could be due to a bad CUDA- or {} "
                    "installation.".format(policy_config["framework"],
                                           policy_config["framework"]))

        self.multiagent: bool = set(
            self.policy_map.keys()) != {DEFAULT_POLICY_ID}
        if self.multiagent and self.env is not None:
            if not ((isinstance(self.env, MultiAgentEnv)
                     or isinstance(self.env, ExternalMultiAgentEnv))
                    or isinstance(self.env, BaseEnv)):
                raise ValueError(
                    "Have multiple policies {}, but the env ".format(
                        self.policy_map) +
                    "{} is not a subclass of BaseEnv, MultiAgentEnv or "
                    "ExternalMultiAgentEnv?".format(self.env))

        self.filters: Dict[PolicyID, Filter] = {
            policy_id: get_filter(self.observation_filter,
                                  policy.observation_space.shape)
            for (policy_id, policy) in self.policy_map.items()
        }
        if self.worker_index == 0:
            logger.info("Built filter map: {}".format(self.filters))

        self.num_envs: int = num_envs

        if self.env is None:
            self.async_env = None
        elif "custom_vector_env" in policy_config:
            custom_vec_wrapper = policy_config["custom_vector_env"]
            self.async_env = custom_vec_wrapper(self.env)
        else:
            # Always use vector env for consistency even if num_envs = 1.
            self.async_env: BaseEnv = BaseEnv.to_base_env(
                self.env,
                make_env=make_env,
                num_envs=num_envs,
                remote_envs=remote_worker_envs,
                remote_env_batch_wait_ms=remote_env_batch_wait_ms,
                policy_config=policy_config,
            )

        # `truncate_episodes`: Allow a batch to contain more than one episode
        # (fragments) and always make the batch `rollout_fragment_length`
        # long.
        if self.batch_mode == "truncate_episodes":
            pack = True
        # `complete_episodes`: Never cut episodes and sampler will return
        # exactly one (complete) episode per poll.
        elif self.batch_mode == "complete_episodes":
            rollout_fragment_length = float("inf")
            pack = False
        else:
            raise ValueError("Unsupported batch mode: {}".format(
                self.batch_mode))

        # Create the IOContext for this worker.
        self.io_context: IOContext = IOContext(log_dir, policy_config,
                                               worker_index, self)
        self.reward_estimators: List[OffPolicyEstimator] = []
        for method in input_evaluation:
            if method == "simulation":
                logger.warning(
                    "Requested 'simulation' input evaluation method: "
                    "will discard all sampler outputs and keep only metrics.")
                sample_async = True
            elif method == "is":
                ise = ImportanceSamplingEstimator.create(self.io_context)
                self.reward_estimators.append(ise)
            elif method == "wis":
                wise = WeightedImportanceSamplingEstimator.create(
                    self.io_context)
                self.reward_estimators.append(wise)
            else:
                raise ValueError(
                    "Unknown evaluation method: {}".format(method))

        render = False
        if policy_config.get("render_env") is True and \
                (num_workers == 0 or worker_index == 1):
            render = True

        if self.env is None:
            self.sampler = None
        elif sample_async:
            self.sampler = AsyncSampler(
                worker=self,
                env=self.async_env,
                clip_rewards=clip_rewards,
                rollout_fragment_length=rollout_fragment_length,
                count_steps_by=count_steps_by,
                callbacks=self.callbacks,
                horizon=episode_horizon,
                multiple_episodes_in_batch=pack,
                tf_sess=self.tf_sess,
                normalize_actions=normalize_actions,
                clip_actions=clip_actions,
                blackhole_outputs="simulation" in input_evaluation,
                soft_horizon=soft_horizon,
                no_done_at_end=no_done_at_end,
                observation_fn=observation_fn,
                sample_collector_class=policy_config.get(
                    "sample_collector_class"),
                render=render,
            )
            # Start the Sampler thread.
            self.sampler.start()
        else:
            self.sampler = SyncSampler(
                worker=self,
                env=self.async_env,
                clip_rewards=clip_rewards,
                rollout_fragment_length=rollout_fragment_length,
                count_steps_by=count_steps_by,
                callbacks=self.callbacks,
                horizon=episode_horizon,
                multiple_episodes_in_batch=pack,
                tf_sess=self.tf_sess,
                normalize_actions=normalize_actions,
                clip_actions=clip_actions,
                soft_horizon=soft_horizon,
                no_done_at_end=no_done_at_end,
                observation_fn=observation_fn,
                sample_collector_class=policy_config.get(
                    "sample_collector_class"),
                render=render,
            )

        self.input_reader: InputReader = input_creator(self.io_context)
        self.output_writer: OutputWriter = output_creator(self.io_context)

        logger.debug(
            "Created rollout worker with env {} ({}), policies {}".format(
                self.async_env, self.env, self.policy_map))

    @DeveloperAPI
    def sample(self) -> SampleBatchType:
        """Returns a batch of experience sampled from this worker.

        This method must be implemented by subclasses.

        Returns:
            SampleBatchType: A columnar batch of experiences (e.g., tensors).

        Examples:
            >>> print(worker.sample())
            SampleBatch({"obs": [1, 2, 3], "action": [0, 1, 0], ...})
        """

        if self.fake_sampler and self.last_batch is not None:
            return self.last_batch
        elif self.input_reader is None:
            raise ValueError("RolloutWorker has no `input_reader` object! "
                             "Cannot call `sample()`.")

        if log_once("sample_start"):
            logger.info("Generating sample batch of size {}".format(
                self.rollout_fragment_length))

        batches = [self.input_reader.next()]
        steps_so_far = batches[0].count if \
            self.count_steps_by == "env_steps" else \
            batches[0].agent_steps()

        # In truncate_episodes mode, never pull more than 1 batch per env.
        # This avoids over-running the target batch size.
        if self.batch_mode == "truncate_episodes":
            max_batches = self.num_envs
        else:
            max_batches = float("inf")

        while (steps_so_far < self.rollout_fragment_length
               and len(batches) < max_batches):
            batch = self.input_reader.next()
            steps_so_far += batch.count if \
                self.count_steps_by == "env_steps" else \
                batch.agent_steps()
            batches.append(batch)
        batch = batches[0].concat_samples(batches) if len(batches) > 1 else \
            batches[0]

        self.callbacks.on_sample_end(worker=self, samples=batch)

        # Always do writes prior to compression for consistency and to allow
        # for better compression inside the writer.
        self.output_writer.write(batch)

        # Do off-policy estimation, if needed.
        if self.reward_estimators:
            for sub_batch in batch.split_by_episode():
                for estimator in self.reward_estimators:
                    estimator.process(sub_batch)

        if log_once("sample_end"):
            logger.info("Completed sample batch:\n\n{}\n".format(
                summarize(batch)))

        if self.compress_observations == "bulk":
            batch.compress(bulk=True)
        elif self.compress_observations:
            batch.compress()

        if self.fake_sampler:
            self.last_batch = batch
        return batch

    @DeveloperAPI
    @ray.method(num_returns=2)
    def sample_with_count(self) -> Tuple[SampleBatchType, int]:
        """Same as sample() but returns the count as a separate future."""
        batch = self.sample()
        return batch, batch.count

    @DeveloperAPI
    def get_weights(self,
                    policies: List[PolicyID] = None) -> (ModelWeights, dict):
        """Returns the model weights of this worker.

        Returns:
            object: weights that can be set on another worker.
            info: dictionary of extra metadata.

        Examples:
            >>> weights = worker.get_weights()
        """
        if policies is None:
            policies = self.policy_map.keys()
        return {
            pid: policy.get_weights()
            for pid, policy in self.policy_map.items() if pid in policies
        }

    @DeveloperAPI
    def set_weights(self, weights: ModelWeights,
                    global_vars: dict = None) -> None:
        """Sets the model weights of this worker.

        Examples:
            >>> weights = worker.get_weights()
            >>> worker.set_weights(weights)
        """
        for pid, w in weights.items():
            self.policy_map[pid].set_weights(w)
        if global_vars:
            self.set_global_vars(global_vars)

    @DeveloperAPI
    def compute_gradients(
            self, samples: SampleBatchType) -> Tuple[ModelGradients, dict]:
        """Returns a gradient computed w.r.t the specified samples.

        Returns:
            (grads, info): A list of gradients that can be applied on a
            compatible worker. In the multi-agent case, returns a dict
            of gradients keyed by policy ids. An info dictionary of
            extra metadata is also returned.

        Examples:
            >>> batch = worker.sample()
            >>> grads, info = worker.compute_gradients(samples)
        """
        if log_once("compute_gradients"):
            logger.info("Compute gradients on:\n\n{}\n".format(
                summarize(samples)))
        if isinstance(samples, MultiAgentBatch):
            grad_out, info_out = {}, {}
            if self.tf_sess is not None:
                builder = TFRunBuilder(self.tf_sess, "compute_gradients")
                for pid, batch in samples.policy_batches.items():
                    if pid not in self.policies_to_train:
                        continue
                    grad_out[pid], info_out[pid] = (
                        self.policy_map[pid]._build_compute_gradients(
                            builder, batch))
                grad_out = {k: builder.get(v) for k, v in grad_out.items()}
                info_out = {k: builder.get(v) for k, v in info_out.items()}
            else:
                for pid, batch in samples.policy_batches.items():
                    if pid not in self.policies_to_train:
                        continue
                    grad_out[pid], info_out[pid] = (
                        self.policy_map[pid].compute_gradients(batch))
        else:
            grad_out, info_out = (
                self.policy_map[DEFAULT_POLICY_ID].compute_gradients(samples))
        info_out["batch_count"] = samples.count
        if log_once("grad_out"):
            logger.info("Compute grad info:\n\n{}\n".format(
                summarize(info_out)))
        return grad_out, info_out

    @DeveloperAPI
    def apply_gradients(self, grads: ModelGradients) -> Dict[PolicyID, Any]:
        """Applies the given gradients to this worker's weights.

        Examples:
            >>> samples = worker.sample()
            >>> grads, info = worker.compute_gradients(samples)
            >>> worker.apply_gradients(grads)
        """
        if log_once("apply_gradients"):
            logger.info("Apply gradients:\n\n{}\n".format(summarize(grads)))
        if isinstance(grads, dict):
            if self.tf_sess is not None:
                builder = TFRunBuilder(self.tf_sess, "apply_gradients")
                outputs = {
                    pid: self.policy_map[pid]._build_apply_gradients(
                        builder, grad)
                    for pid, grad in grads.items()
                }
                return {k: builder.get(v) for k, v in outputs.items()}
            else:
                return {
                    pid: self.policy_map[pid].apply_gradients(g)
                    for pid, g in grads.items()
                }
        else:
            return self.policy_map[DEFAULT_POLICY_ID].apply_gradients(grads)

    @DeveloperAPI
    def learn_on_batch(self, samples: SampleBatchType) -> dict:
        """Update policies based on the given batch.

        This is the equivalent to apply_gradients(compute_gradients(samples)),
        but can be optimized to avoid pulling gradients into CPU memory.

        Returns:
            info: dictionary of extra metadata from compute_gradients().

        Examples:
            >>> batch = worker.sample()
            >>> worker.learn_on_batch(samples)
        """
        if log_once("learn_on_batch"):
            logger.info(
                "Training on concatenated sample batches:\n\n{}\n".format(
                    summarize(samples)))
        if isinstance(samples, MultiAgentBatch):
            info_out = {}
            to_fetch = {}
            if self.tf_sess is not None:
                builder = TFRunBuilder(self.tf_sess, "learn_on_batch")
            else:
                builder = None
            for pid, batch in samples.policy_batches.items():
                if pid not in self.policies_to_train:
                    continue
                # Decompress SampleBatch, in case some columns are compressed.
                batch.decompress_if_needed()
                policy = self.policy_map[pid]
                if builder and hasattr(policy, "_build_learn_on_batch"):
                    to_fetch[pid] = policy._build_learn_on_batch(
                        builder, batch)
                else:
                    info_out[pid] = policy.learn_on_batch(batch)
            info_out.update({k: builder.get(v) for k, v in to_fetch.items()})
        else:
            info_out = {
                DEFAULT_POLICY_ID: self.policy_map[DEFAULT_POLICY_ID]
                .learn_on_batch(samples)
            }
        if log_once("learn_out"):
            logger.debug("Training out:\n\n{}\n".format(summarize(info_out)))
        return info_out

    def sample_and_learn(self, expected_batch_size: int, num_sgd_iter: int,
                         sgd_minibatch_size: str,
                         standardize_fields: List[str]) -> Tuple[dict, int]:
        """Sample and batch and learn on it.

        This is typically used in combination with distributed allreduce.

        Args:
            expected_batch_size (int): Expected number of samples to learn on.
            num_sgd_iter (int): Number of SGD iterations.
            sgd_minibatch_size (int): SGD minibatch size.
            standardize_fields (list): List of sample fields to normalize.

        Returns:
            info: dictionary of extra metadata from learn_on_batch().
            count: number of samples learned on.
        """
        batch = self.sample()
        assert batch.count == expected_batch_size, \
            ("Batch size possibly out of sync between workers, expected:",
             expected_batch_size, "got:", batch.count)
        logger.info("Executing distributed minibatch SGD "
                    "with epoch size {}, minibatch size {}".format(
                        batch.count, sgd_minibatch_size))
        info = do_minibatch_sgd(batch, self.policy_map, self, num_sgd_iter,
                                sgd_minibatch_size, standardize_fields)
        return info, batch.count

    @DeveloperAPI
    def get_metrics(self) -> List[Union[RolloutMetrics, OffPolicyEstimate]]:
        """Returns a list of new RolloutMetric objects from evaluation."""

        # Get metrics from sampler (if any).
        if self.sampler is not None:
            out = self.sampler.get_metrics()
        else:
            out = []
        # Get metrics from our reward-estimators (if any).
        for m in self.reward_estimators:
            out.extend(m.get_metrics())

        return out

    @DeveloperAPI
    def foreach_env(self, func: Callable[[BaseEnv], T]) -> List[T]:
        """Apply the given function to each underlying env instance."""

        if self.async_env is None:
            return []

        envs = self.async_env.get_unwrapped()
        if not envs:
            return [func(self.async_env)]
        else:
            return [func(e) for e in envs]

    @DeveloperAPI
    def foreach_env_with_context(
            self, func: Callable[[BaseEnv, EnvContext], T]) -> List[T]:
        """Apply the given function to each underlying env instance."""

        if self.async_env is None:
            return []

        envs = self.async_env.get_unwrapped()
        if not envs:
            return [func(self.async_env, self.env_context)]
        else:
            ret = []
            for i, e in enumerate(envs):
                ctx = self.env_context.copy_with_overrides(vector_index=i)
                ret.append(func(e, ctx))
            return ret

    @DeveloperAPI
    def get_policy(
            self, policy_id: Optional[PolicyID] = DEFAULT_POLICY_ID) -> Policy:
        """Return policy for the specified id, or None.

        Args:
            policy_id (str): id of policy to return.
        """

        return self.policy_map.get(policy_id)

    @DeveloperAPI
    def add_policy(
            self,
            *,
            policy_id: PolicyID,
            policy_cls: Type[Policy],
            observation_space: Optional[gym.spaces.Space] = None,
            action_space: Optional[gym.spaces.Space] = None,
            config: Optional[PartialTrainerConfigDict] = None,
            policy_mapping_fn: Optional[Callable[
                [AgentID, "MultiAgentEpisode"], PolicyID]] = None,
            policies_to_train: Optional[List[PolicyID]] = None,
    ) -> Policy:
        """Adds a new policy to this RolloutWorker.

        Args:
            policy_id (Optional[PolicyID]): ID of the policy to add.
            policy_cls (Type[Policy]): The Policy class to use for
                constructing the new Policy.
            observation_space (Optional[gym.spaces.Space]): The observation
                space of the policy to add.
            action_space (Optional[gym.spaces.Space]): The action space
                of the policy to add.
            config (Optional[PartialTrainerConfigDict]): The config overrides
                for the policy to add.
            policy_mapping_fn (Optional[Callable[[AgentID, MultiAgentEpisode],
                PolicyID]]): An optional (updated) policy mapping function to
                use from here on. Note that already ongoing episodes will not
                change their mapping but will use the old mapping till the
                end of the episode.
            policies_to_train (Optional[List[PolicyID]]): An optional list of
                policy IDs to be trained. If None, will keep the existing list
                in place. Policies, whose IDs are not in the list will not be
                updated.

        Returns:
            Policy: The newly added policy (the copy that got added to the
                local worker).
        """
        if policy_id in self.policy_map:
            raise ValueError(f"Policy ID '{policy_id}' already in policy map!")
        policy_dict = {
            policy_id: (policy_cls, observation_space, action_space, config)
        }
        add_map, add_prep = self._build_policy_map(policy_dict,
                                                   self.policy_config)
        new_policy = add_map[policy_id]

        self.policy_map.update(add_map)
        self.preprocessors.update(add_prep)
        self.filters[policy_id] = get_filter(
            self.observation_filter, new_policy.observation_space.shape)

        self.set_policy_mapping_fn(policy_mapping_fn)
        self.set_policies_to_train(policies_to_train)

        return new_policy

    @DeveloperAPI
    def remove_policy(
            self,
            *,
            policy_id: PolicyID = DEFAULT_POLICY_ID,
            policy_mapping_fn: Optional[Callable[[AgentID], PolicyID]] = None,
            policies_to_train: Optional[List[PolicyID]] = None,
    ):
        """Removes a policy from this RolloutWorker.

        Args:
            policy_id (Optional[PolicyID]): ID of the policy to be removed.
            policy_mapping_fn (Optional[Callable[[AgentID], PolicyID]]): An
                optional (updated) policy mapping function to use from here on.
                Note that already ongoing episodes will not change their
                mapping but will use the old mapping till the end of the
                episode.
            policies_to_train (Optional[List[PolicyID]]): An optional list of
                policy IDs to be trained. If None, will keep the existing list
                in place. Policies, whose IDs are not in the list will not be
                updated.
        """
        if policy_id not in self.policy_map:
            raise ValueError(f"Policy ID '{policy_id}' not in policy map!")
        del self.policy_map[policy_id]
        del self.preprocessors[policy_id]
        self.set_policy_mapping_fn(policy_mapping_fn)
        self.set_policies_to_train(policies_to_train)

    @DeveloperAPI
    def set_policy_mapping_fn(
            self,
            policy_mapping_fn: Optional[Callable[
                [AgentID, "MultiAgentEpisode"], PolicyID]] = None,
    ):
        """Sets `self.policy_mapping_fn` to a new callable (if provided).

        Args:
            policy_mapping_fn (Optional[Callable[[AgentID], PolicyID]]): The
                new mapping function to use. If None, will keep the existing
                mapping function in place.
        """
        if policy_mapping_fn is not None:
            self.policy_mapping_fn = policy_mapping_fn
            if not callable(self.policy_mapping_fn):
                raise ValueError("`policy_mapping_fn` must be a callable!")

    @DeveloperAPI
    def set_policies_to_train(
            self, policies_to_train: Optional[List[PolicyID]] = None):
        """Sets `self.policies_to_train` to a new list of PolicyIDs.

        Args:
            policies_to_train (Optional[List[PolicyID]]): The new
                list of policy IDs to train with. If None, will keep the
                existing list in place.
        """
        if policies_to_train is not None:
            self.policies_to_train = policies_to_train

    @DeveloperAPI
    def for_policy(self,
                   func: Callable[[Policy], T],
                   policy_id: Optional[PolicyID] = DEFAULT_POLICY_ID,
                   **kwargs) -> T:
        """Apply the given function to the specified policy."""

        return func(self.policy_map[policy_id], **kwargs)

    @DeveloperAPI
    def foreach_policy(self, func: Callable[[Policy, PolicyID], T],
                       **kwargs) -> List[T]:
        """Apply the given function to each (policy, policy_id) tuple."""

        return [
            func(policy, pid, **kwargs)
            for pid, policy in self.policy_map.items()
        ]

    @DeveloperAPI
    def foreach_trainable_policy(self, func: Callable[[Policy, PolicyID], T],
                                 **kwargs) -> List[T]:
        """
        Applies the given function to each (policy, policy_id) tuple, which
        can be found in `self.policies_to_train`.

        Args:
            func (callable): A function - taking a Policy and its ID - that is
                called on all Policies within `self.policies_to_train`.

        Returns:
            List[any]: The list of n return values of all
                `func([policy], [ID])`-calls.
        """
        return [
            func(policy, pid, **kwargs)
            for pid, policy in self.policy_map.items()
            if pid in self.policies_to_train
        ]

    @DeveloperAPI
    def sync_filters(self, new_filters: dict) -> None:
        """Changes self's filter to given and rebases any accumulated delta.

        Args:
            new_filters (dict): Filters with new state to update local copy.
        """
        assert all(k in new_filters for k in self.filters)
        for k in self.filters:
            self.filters[k].sync(new_filters[k])

    @DeveloperAPI
    def get_filters(self, flush_after: bool = False) -> dict:
        """Returns a snapshot of filters.

        Args:
            flush_after (bool): Clears the filter buffer state.

        Returns:
            return_filters (dict): Dict for serializable filters
        """
        return_filters = {}
        for k, f in self.filters.items():
            return_filters[k] = f.as_serializable()
            if flush_after:
                f.clear_buffer()
        return return_filters

    @DeveloperAPI
    def save(self) -> bytes:
        filters = self.get_filters(flush_after=True)
        state = {
            pid: self.policy_map[pid].get_state()
            for pid in self.policy_map
        }
        return pickle.dumps({"filters": filters, "state": state})

    @DeveloperAPI
    def restore(self, objs: bytes) -> None:
        objs = pickle.loads(objs)
        self.sync_filters(objs["filters"])
        for pid, state in objs["state"].items():
            if pid not in self.policy_map:
                logger.warning(
                    f"pid={pid} not found in policy_map! It was probably added"
                    " on-the-fly and is not part of the static `config."
                    "multiagent.policies` dict. Ignoring it for now.")
                continue
            self.policy_map[pid].set_state(state)

    @DeveloperAPI
    def set_global_vars(self, global_vars: dict) -> None:
        self.foreach_policy(lambda p, _: p.on_global_var_update(global_vars))
        self.global_vars = global_vars

    @DeveloperAPI
    def get_global_vars(self) -> dict:
        return self.global_vars

    @DeveloperAPI
    def export_policy_model(self,
                            export_dir: str,
                            policy_id: PolicyID = DEFAULT_POLICY_ID):
        self.policy_map[policy_id].export_model(export_dir)

    @DeveloperAPI
    def import_policy_model_from_h5(self,
                                    import_file: str,
                                    policy_id: PolicyID = DEFAULT_POLICY_ID):
        self.policy_map[policy_id].import_model_from_h5(import_file)

    @DeveloperAPI
    def export_policy_checkpoint(self,
                                 export_dir: str,
                                 filename_prefix: str = "model",
                                 policy_id: PolicyID = DEFAULT_POLICY_ID):
        self.policy_map[policy_id].export_checkpoint(export_dir,
                                                     filename_prefix)

    @DeveloperAPI
    def stop(self) -> None:
        if self.env is not None:
            self.async_env.stop()

    @DeveloperAPI
    def creation_args(self) -> dict:
        """Returns the args used to create this worker."""
        return self._original_kwargs

    @DeveloperAPI
    def get_host(self) -> str:
        """Returns the hostname of the process running this evaluator."""

        return platform.node()

    @DeveloperAPI
    def apply(self, func: Callable[["RolloutWorker"], T], *args) -> T:
        """Apply the given function to this rollout worker instance."""

        return func(self, *args)

    def _build_policy_map(
            self, policy_dict: MultiAgentPolicyConfigDict,
            policy_config: TrainerConfigDict
    ) -> Tuple[Dict[PolicyID, Policy], Dict[PolicyID, Preprocessor]]:
        policy_map = {}
        preprocessors = {}
        for name, (cls, obs_space, act_space,
                   conf) in sorted(policy_dict.items()):
            logger.debug("Creating policy for {}".format(name))
            merged_conf = merge_dicts(policy_config, conf)
            merged_conf["num_workers"] = self.num_workers
            merged_conf["worker_index"] = self.worker_index
            if self.preprocessing_enabled:
                preprocessor = ModelCatalog.get_preprocessor_for_space(
                    obs_space, merged_conf.get("model"))
                preprocessors[name] = preprocessor
                obs_space = preprocessor.observation_space
            else:
                preprocessors[name] = NoPreprocessor(obs_space)

            if isinstance(obs_space, (gym.spaces.Dict, gym.spaces.Tuple)):
                raise ValueError(
                    "Found raw Tuple|Dict space as input to policy. "
                    "Please preprocess these observations with a "
                    "Tuple|DictFlatteningPreprocessor.")
            # Tf.
            framework = policy_config.get("framework", "tf")
            if framework in ["tf2", "tf", "tfe"]:
                assert tf1
                if framework in ["tf2", "tfe"]:
                    assert tf1.executing_eagerly()
                    if hasattr(cls, "as_eager"):
                        cls = cls.as_eager()
                        if policy_config.get("eager_tracing"):
                            cls = cls.with_tracing()
                    elif not issubclass(cls, TFPolicy):
                        pass  # could be some other type of policy
                    else:
                        raise ValueError("This policy does not support eager "
                                         "execution: {}".format(cls))
                scope = name + (("_wk" + str(self.worker_index))
                                if self.worker_index else "")
                with tf1.variable_scope(scope):
                    policy_map[name] = cls(obs_space, act_space, merged_conf)
            # non-tf.
            else:
                policy_map[name] = cls(obs_space, act_space, merged_conf)

        if self.worker_index == 0:
            logger.info("Built policy map: {}".format(policy_map))
            logger.info("Built preprocessor map: {}".format(preprocessors))
        return policy_map, preprocessors

    def setup_torch_data_parallel(self, url: str, world_rank: int,
                                  world_size: int, backend: str) -> None:
        """Join a torch process group for distributed SGD."""

        logger.info("Joining process group, url={}, world_rank={}, "
                    "world_size={}, backend={}".format(url, world_rank,
                                                       world_size, backend))
        torch.distributed.init_process_group(
            backend=backend,
            init_method=url,
            rank=world_rank,
            world_size=world_size)

        for pid, policy in self.policy_map.items():
            if not isinstance(policy, TorchPolicy):
                raise ValueError(
                    "This policy does not support torch distributed", policy)
            policy.distributed_world_size = world_size

    def get_node_ip(self) -> str:
        """Returns the IP address of the current node."""
        return ray.util.get_node_ip_address()

    def find_free_port(self) -> int:
        """Finds a free port on the current node."""
        from ray.util.sgd import utils
        return utils.find_free_port()

    def __del__(self):
        if hasattr(self, "sampler") and isinstance(self.sampler, AsyncSampler):
            self.sampler.shutdown = True


def _validate_and_canonicalize(
        policy: Union[Type[Policy], MultiAgentPolicyConfigDict],
        env: Optional[EnvType],
        spaces: Optional[Dict[PolicyID, Tuple[gym.spaces.Space,
                                              gym.spaces.Space]]],
        policy_config: Optional[PartialTrainerConfigDict],
) -> MultiAgentPolicyConfigDict:

    if isinstance(policy, dict):
        _validate_multiagent_config(policy)
        return policy
    elif not issubclass(policy, Policy):
        raise ValueError(f"`policy` ({policy}) must be a rllib.Policy class!")
    else:
        if (isinstance(env, MultiAgentEnv)
                and not hasattr(env, "observation_space")):
            raise ValueError(
                "MultiAgentEnv must have observation_space defined if run "
                "in a single-agent configuration.")
        if env is not None:
            return {
                DEFAULT_POLICY_ID: (policy, env.observation_space,
                                    env.action_space, {})
            }

        if spaces is None:
            if "action_space" not in policy_config or \
                    "observation_space" not in policy_config:
                raise ValueError(
                    "If no env given, must provide obs/action spaces either "
                    "in the `multiagent.policies` dict or under "
                    "`config.[observation|action]_space`!")
            spaces = {
                DEFAULT_POLICY_ID: (policy_config["observation_space"],
                                    policy_config["action_space"])
            }
        return {
            DEFAULT_POLICY_ID: (policy, spaces[DEFAULT_POLICY_ID][0],
                                spaces[DEFAULT_POLICY_ID][1], {})
        }


def _validate_multiagent_config(policy: MultiAgentPolicyConfigDict,
                                allow_none_graph: bool = False) -> None:
    # Loop through all policy definitions in multi-agent policie
    for k, v in policy.items():
        if not isinstance(k, str):
            raise ValueError("Policy key must be str, got {}!".format(k))
        if not isinstance(v, (tuple, list)) or len(v) != 4:
            raise ValueError(
                "policy values must be tuples/lists of "
                "(cls or None, obs_space, action_space, config), got {}".
                format(v))
        if allow_none_graph and v[0] is None:
            pass
        elif not issubclass(v[0], Policy):
            raise ValueError("policy tuple value 0 must be a rllib.Policy "
                             "class or None, got {}".format(v[0]))
        if not isinstance(v[1], gym.Space):
            raise ValueError(
                "policy tuple value 1 (observation_space) must be a "
                "gym.Space, got {}".format(type(v[1])))
        if not isinstance(v[2], gym.Space):
            raise ValueError("policy tuple value 2 (action_space) must be a "
                             "gym.Space, got {}".format(type(v[2])))
        if not isinstance(v[3], dict):
            raise ValueError("policy tuple value 3 (config) must be a dict, "
                             "got {}".format(type(v[3])))


def _validate_env(env: Any) -> EnvType:
    # Allow this as a special case (assumed gym.Env).
    if hasattr(env, "observation_space") and hasattr(env, "action_space"):
        return env

    allowed_types = [gym.Env, MultiAgentEnv, ExternalEnv, VectorEnv, BaseEnv]
    if not any(isinstance(env, tpe) for tpe in allowed_types):
        raise ValueError(
            "Returned env should be an instance of gym.Env, MultiAgentEnv, "
            "ExternalEnv, VectorEnv, or BaseEnv. The provided env creator "
            "function returned {} ({}).".format(env, type(env)))
    return env


def _has_tensorflow_graph(policy_dict: MultiAgentPolicyConfigDict) -> bool:
    for policy, _, _, _ in policy_dict.values():
        if issubclass(policy, TFPolicy):
            return True
    return False
