// Copyright 2021 The Ray Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "ray/raylet/scheduling/scheduling_policy.h"

#include <algorithm>
#include <cstdint>
#include <functional>

#include "absl/types/optional.h"
#include "ray/common/ray_config.h"

namespace ray {

namespace raylet_scheduling_policy {
namespace {

bool IsGPURequest(const ResourceRequest &resource_request) {
  if (resource_request.predefined_resources.size() <= GPU) {
    return false;
  }
  return resource_request.predefined_resources[GPU] > 0;
}

bool DoesNodeHaveGPUs(const NodeResources &resources) {
  if (resources.predefined_resources.size() <= GPU) {
    return false;
  }
  return resources.predefined_resources[GPU].total > 0;
}
}  // namespace

struct NodeInfo {
  bool is_feasible = false;
  bool is_available = false;
  float critical_resource_utilization = 1.0f;
  float weight = 0.0f;
  friend std::ostream &operator<<(std::ostream &os, const NodeInfo &info) {
    return os << (info.is_feasible ? "feasible" : "!feasible") << ","
              << (info.is_available ? "available" : "!available") << ","
              << "critical: " << info.critical_resource_utilization
              << "\tweight: " << info.weight;
  }
};

#define _DEBUG DEBUG
NodeInfo GetNodeInfo(const Node &node, const ResourceRequest &resource_request) {
  NodeInfo info;
  info.is_feasible = node.GetLocalView().IsFeasible(resource_request);
  if (!info.is_feasible) {
    return info;
  }
  info.is_available = node.GetLocalView().IsAvailable(resource_request, true);
  if (!info.is_available) {
    return info;
  }
  // TODO: consider taking resource_request into account when computing
  // CalculateCriticalResourceUtilization.
  info.critical_resource_utilization =
      node.GetLocalView().CalculateCriticalResourceUtilization();

  absl::optional<float> min;

  for (const auto &i : {CPU, MEM, OBJECT_STORE_MEM, GPU}) {
    if (i >= node.GetLocalView().predefined_resources.size()) {
      break;
    }
    if (i >= resource_request.predefined_resources.size()) {
      break;
    }
    const auto &capacity = node.GetLocalView().predefined_resources[i];

    RAY_LOG(_DEBUG) << "n: " << i << "\t" << capacity.total << " "
                    << resource_request.predefined_resources[i] << " "
                    << (min ? min.value() : -1);
    if (capacity.total == 0) {
      continue;
    }
    if (resource_request.predefined_resources[i] == 0) {
      continue;
    }
    float r =
        capacity.available.Double() / resource_request.predefined_resources[i].Double();
    RAY_LOG(_DEBUG) << "r: " << r << " " << min.has_value() << " "
                    << (!min.has_value() || *min > r);
    if (!min.has_value() || *min > r) {
      min = r;
    }
  }

  if (min.has_value()) {
    info.weight = min.value();
  }
  return info;
}

int64_t NewHybridPolicyWithFilter(const ResourceRequest &resource_request,
                                  const int64_t local_node_id,
                                  const absl::flat_hash_map<int64_t, Node> &nodes,
                                  float spread_threshold, bool force_spillback,
                                  bool require_available, NodeFilter node_filter) {
  // Similarly to HybridPolicy, prefer scheduling on local node,
  // perform packing in case of low utilization.
  // Perform weighted random scheduling otherwise.

  auto predicate = [node_filter](const auto &node) {
    if (node_filter == NodeFilter::kAny) {
      return true;
    }
    const bool has_gpu = DoesNodeHaveGPUs(node);
    if (node_filter == NodeFilter::kGPU) {
      return has_gpu;
    }
    RAY_CHECK(node_filter == NodeFilter::kCPUOnly);
    return !has_gpu;
  };

  const auto local_node_it = nodes.find(local_node_id);
  RAY_CHECK(local_node_it != nodes.end());
  const auto &local_node = local_node_it->second;
  const auto local_info = GetNodeInfo(local_node, resource_request);
  RAY_LOG(_DEBUG) << "Local node: " << local_node_id << " " << local_info;

  if (!force_spillback && predicate(local_node.GetLocalView()) &&
      local_info.is_feasible && local_info.is_available &&
      local_info.critical_resource_utilization < spread_threshold) {
    return local_node_id;
  }

  int64_t feasible_node_id = -1;
  if (local_info.is_feasible && !force_spillback &&
      predicate(local_node.GetLocalView())) {
    RAY_LOG(_DEBUG) << "feasible id: " << feasible_node_id;
    feasible_node_id = local_node_id;
  }

  std::vector<std::pair<float, int64_t>> ut_node_id;
  ut_node_id.reserve(nodes.size());
  if (local_info.is_feasible && local_info.is_available && !force_spillback &&
      predicate(local_node.GetLocalView())) {
    ut_node_id.push_back({local_info.weight, local_node_id});
  }
  std::vector<int64_t> ids;
  ids.reserve(nodes.size());
  for (const auto &pair : nodes) {
    if (pair.first != local_node_id && predicate(pair.second.GetLocalView())) {
      ids.push_back(pair.first);
    }
  }
  std::sort(ids.begin(), ids.end());

  for (auto id_it = ids.begin(); id_it != ids.end(); ++id_it) {
    const auto node_id = *id_it;
    const auto node_it = nodes.find(node_id);
    RAY_CHECK(node_it != nodes.end());
    const auto &node = node_it->second;
    const auto info = GetNodeInfo(node, resource_request);

    RAY_LOG(_DEBUG) << "node " << node_id << " " << info
                    << " spread_threshold=" << spread_threshold;

    if (!info.is_feasible) {
      continue;
    }
    if (feasible_node_id == -1) {
      RAY_LOG(_DEBUG) << "feasible id: " << feasible_node_id;
      feasible_node_id = node_id;
    }
    if (!info.is_available) {
      continue;
    }
    if (info.critical_resource_utilization < spread_threshold) {
      return node_id;
    }
    ut_node_id.push_back({info.weight, node_id});
  }

  if (ut_node_id.empty()) {
    if (require_available) {
      return -1;
    }
    return feasible_node_id;
  }
  // Not needed.
  // std::sort(ut_node_id.begin(), ut_node_id.end());

  float sum = 0;
  for (auto it = ut_node_id.begin(); it != ut_node_id.end(); ++it) {
    const float tmp = it->first;
    it->first = sum;
    sum += tmp;
  }
  static thread_local std::default_random_engine gen;
  std::uniform_real_distribution<double> distribution(0, sum);
  const float w = distribution(gen);
  auto lb = std::lower_bound(ut_node_id.begin(), ut_node_id.end(),
                             std::make_pair(w, int64_t()));
  if (lb == ut_node_id.end()) {
    // Just in case.
    --lb;
  }

  const int64_t random_node_id = lb->second;
  RAY_LOG(DEBUG) << "random_node_id=" << random_node_id << " w:" << w << " max:" << sum;
  return random_node_id;
}

int64_t HybridPolicyWithFilter(const ResourceRequest &resource_request,
                               const int64_t local_node_id,
                               const absl::flat_hash_map<int64_t, Node> &nodes,
                               float spread_threshold, bool force_spillback,
                               bool require_available, NodeFilter node_filter) {
  if (!RayConfig::instance().scheduler_old()) {
    return NewHybridPolicyWithFilter(resource_request, local_node_id, nodes,
                                     spread_threshold, force_spillback, require_available,
                                     node_filter);
  }
  // Step 1: Generate the traversal order. We guarantee that the first node is local, to
  // encourage local scheduling. The rest of the traversal order should be globally
  // consistent, to encourage using "warm" workers.
  std::vector<int64_t> round;
  round.reserve(nodes.size());
  const auto local_it = nodes.find(local_node_id);
  RAY_CHECK(local_it != nodes.end());

  auto predicate = [node_filter](const auto &node) {
    if (node_filter == NodeFilter::kAny) {
      return true;
    }
    const bool has_gpu = DoesNodeHaveGPUs(node);
    if (node_filter == NodeFilter::kGPU) {
      return has_gpu;
    }
    RAY_CHECK(node_filter == NodeFilter::kCPUOnly);
    return !has_gpu;
  };

  const auto &local_node = local_it->second.GetLocalView();
  // If we should include local node at all, make sure it is at the front of the list
  // so that
  // 1. It's first in traversal order.
  // 2. It's easy to avoid sorting it.
  if (predicate(local_node) && !force_spillback) {
    round.push_back(local_node_id);
  }

  const auto start_index = round.size();
  for (const auto &pair : nodes) {
    if (pair.first != local_node_id && predicate(pair.second.GetLocalView())) {
      round.push_back(pair.first);
    }
  }
  // Sort all the nodes, making sure that if we added the local node in front, it stays in
  // place.
  std::sort(round.begin() + start_index, round.end());

  int64_t best_node_id = -1;
  float best_utilization_score = INFINITY;
  bool best_is_available = false;

  // Step 2: Perform the round robin.
  auto round_it = round.begin();
  for (; round_it != round.end(); round_it++) {
    const auto &node_id = *round_it;
    const auto &it = nodes.find(node_id);
    RAY_CHECK(it != nodes.end());
    const auto &node = it->second;
    if (!node.GetLocalView().IsFeasible(resource_request)) {
      continue;
    }

    bool ignore_pull_manager_at_capacity = false;
    if (node_id == local_node_id) {
      // It's okay if the local node's pull manager is at
      // capacity because we will eventually spill the task
      // back from the waiting queue if its args cannot be
      // pulled.
      ignore_pull_manager_at_capacity = true;
    }
    bool is_available = node.GetLocalView().IsAvailable(resource_request,
                                                        ignore_pull_manager_at_capacity);
    RAY_LOG(DEBUG) << "Node " << node_id << " is "
                   << (is_available ? "available" : "not available");
    float critical_resource_utilization =
        node.GetLocalView().CalculateCriticalResourceUtilization();
    if (critical_resource_utilization < spread_threshold) {
      critical_resource_utilization = 0;
    }

    bool update_best_node = false;

    if (is_available) {
      // Always prioritize available nodes over nodes where the task must be queued first.
      if (!best_is_available) {
        update_best_node = true;
      } else if (critical_resource_utilization < best_utilization_score) {
        // Break ties between available nodes by their critical resource utilization.
        update_best_node = true;
      }
    } else if (!best_is_available &&
               critical_resource_utilization < best_utilization_score &&
               !require_available) {
      // Pick the best feasible node by critical resource utilization.
      update_best_node = true;
    }

    if (update_best_node) {
      best_node_id = node_id;
      best_utilization_score = critical_resource_utilization;
      best_is_available = is_available;
    }
  }

  return best_node_id;
}

int64_t HybridPolicy(const ResourceRequest &resource_request, const int64_t local_node_id,
                     const absl::flat_hash_map<int64_t, Node> &nodes,
                     float spread_threshold, bool force_spillback, bool require_available,
                     bool scheduler_avoid_gpu_nodes) {
  if (!scheduler_avoid_gpu_nodes || IsGPURequest(resource_request)) {
    return HybridPolicyWithFilter(resource_request, local_node_id, nodes,
                                  spread_threshold, force_spillback, require_available);
  }

  // Try schedule on CPU-only nodes.
  const auto node_id =
      HybridPolicyWithFilter(resource_request, local_node_id, nodes, spread_threshold,
                             force_spillback, require_available, NodeFilter::kCPUOnly);
  if (node_id != -1) {
    return node_id;
  }
  // Could not schedule on CPU-only nodes, schedule on GPU nodes as a last resort.
  return HybridPolicyWithFilter(resource_request, local_node_id, nodes, spread_threshold,
                                force_spillback, require_available, NodeFilter::kGPU);
}

}  // namespace raylet_scheduling_policy
}  // namespace ray
