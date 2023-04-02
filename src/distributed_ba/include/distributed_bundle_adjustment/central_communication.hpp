#pragma once

#include <ceres/ceres.h>
#include <memory>
#include <thread>

#include "distributed_bundle_adjustment/common.hpp"
#include "distributed_bundle_adjustment/data.hpp"
#include "distributed_bundle_adjustment/optimization.hpp"

namespace dba {

class MasterNode {
 public:
  using FrameDual = PoseDual<kNumIntrinsicParams, kNumDistortionParams>;

 public:
  MasterNode() = delete;
  MasterNode(const int num_workers, DataSharedPtr data_ptr);
  ~MasterNode();

  auto startNodes() -> bool;

 private:
  auto communicationLoop() -> void;

  auto computeFinalResult() -> void;

  const int num_workers_;
  std::thread comm_thread_;
  DataSharedPtr data_ptr_;
  std::unordered_set<uint64_t> outlier_vars_;
  bool has_started_;
};

class WorkerNode {
 public:
  using FrameDual = PoseDual<kNumIntrinsicParams, kNumDistortionParams>;

 public:
  WorkerNode() = delete;
  WorkerNode(const int rank, DataSharedPtr data_ptr);
  ~WorkerNode();

 private:
  auto startNode() -> void;

  auto processLoop() -> void;

  auto communicateData() -> void;

  const int rank_;
  DataSharedPtr data_ptr_;
  OptimizationUniquePtr optimization_ptr_;
  std::thread process_thread_;
};

}  // namespace dba
