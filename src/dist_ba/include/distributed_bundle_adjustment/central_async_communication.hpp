#pragma once

#include <ceres/ceres.h>
#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>

#include "distributed_bundle_adjustment/common.hpp"
#include "distributed_bundle_adjustment/data.hpp"
#include "distributed_bundle_adjustment/optimization.hpp"

namespace dba {

class AsyncMasterNode {
 public:
  using FrameDual = PoseDual<kNumIntrinsicParams, kNumDistortionParams>;

 public:
  AsyncMasterNode() = delete;
  AsyncMasterNode(const int num_workers, DataSharedPtr data_ptr);
  ~AsyncMasterNode();

  auto startNodes() -> bool;

 private:
  auto communicationLoop() -> void;

  auto receiverThread(const uint64_t neigh_id) -> void;

  auto updateVariables() -> void;

  auto communicateUpdates() -> void;

  const int num_workers_;
  std::thread comm_thread_;
  DataSharedPtr data_ptr_;
  std::unordered_set<uint64_t> outlier_variables_;
  std::mutex start_mutex_;
  std::condition_variable start_cv_;
  bool has_started_;

  // Synchronization/counting of incoming variables
  std::mutex data_mutex_;
  std::vector<std::thread> receiver_threads_;
  std::condition_variable incoming_cv_;
  std::unordered_map<uint64_t, std::vector<FrameDual>> incoming_frames_;
  std::unordered_map<uint64_t, std::vector<MapPointDual>> incoming_map_points_;
  bool has_data_;
  int iteration_counter_;
};

class AsyncWorkerNode {
 public:
  using FrameDual = PoseDual<kNumIntrinsicParams, kNumDistortionParams>;

 public:
  AsyncWorkerNode() = delete;
  AsyncWorkerNode(const int num_workers, const int rank,
                  DataSharedPtr data_ptr);
  ~AsyncWorkerNode();

 private:
  auto startNode() -> void;

  auto processLoop() -> void;

  auto communicateData() -> void;

  auto globalCounter() -> void;

  const int num_workers_;
  const int rank_;
  DataSharedPtr data_ptr_;
  OptimizationUniquePtr optimization_ptr_;
  std::thread process_thread_;

  // Tracking of the global state (i.e. iterations)
  std::thread counter_thread_;
  int local_iterations_;
  std::atomic<int> global_iterations_;
  int last_saved_state_;
};

}  // namespace dba
