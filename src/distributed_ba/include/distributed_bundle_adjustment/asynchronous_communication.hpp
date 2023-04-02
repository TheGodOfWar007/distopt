#pragma once

#include <mpi.h>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <stack>
#include <thread>
#include <vector>

#include "distributed_bundle_adjustment/common.hpp"
#include "distributed_bundle_adjustment/data.hpp"
#include "distributed_bundle_adjustment/optimization.hpp"

namespace dba {

class AsynchronousCoordinator {
 public:
  using FrameDual = PoseDual<kNumIntrinsicParams, kNumDistortionParams>;
  using DataSharedPtrVector = std::vector<DataSharedPtr>;
  using DataIdMap = std::unordered_map<uint64_t, DataSharedPtr>;
  using EdgeId = std::pair<uint64_t, uint64_t>;
  template <class DataType>
  using EdgeToDataMap = std::unordered_map<EdgeId, DataType, pair_hash>;

 public:
  AsynchronousCoordinator() = delete;
  AsynchronousCoordinator(const DataSharedPtrVector& data);
  ~AsynchronousCoordinator();

 private:
  auto initializeBuffers() -> void;

  auto mainThread() -> void;

  auto updateCounter() -> bool;

  auto updateBuffers() -> void;

  auto checkAndUpdateBuffer(const EdgeId& id, std::vector<double>& buffer)
      -> bool;

  auto sendUpdates(const uint64_t& node_id) -> void;

  // Useful handles
  int world_size_;

  DataIdMap data_map_;

  // Storage for maintaining the iteration counts
  int* counter_storage_;
  MPI_Win counter_window_;
  std::thread main_thread_;
  int iteration_counter_;

  // Keep a flag array to check whether a node has finished
  int* finish_flags_;
  MPI_Win finish_flag_window_;

  // Storage for indicating whether a node requires data transfer
  int* flag_storage_;
  MPI_Win flag_window_;
  std::vector<int> last_flag_state_;
  EdgeToDataMap<std::vector<double>> data_buffer_map_;
  EdgeToDataMap<MPI_Comm> communicator_map_;
};

class AsynchronousCommunication {
 public:
  using FrameDual = PoseDual<kNumIntrinsicParams, kNumDistortionParams>;

 public:
  AsynchronousCommunication() = delete;
  AsynchronousCommunication(DataSharedPtr data_ptr,
                            DataSharedPtr data_copy_ptr);
  ~AsynchronousCommunication();

 private:
  auto mainThread() -> void;

  auto communicateData() -> void;

  auto updateReceivedDuals() -> void;

  auto updateDuals() -> void;

  auto writeOutStatus() -> void;

  std::thread main_thread_;
  DataSharedPtr data_ptr_;
  DataSharedPtr data_copy_ptr_;
  OptimizationUniquePtr optimization_ptr_;

  bool run_;
  bool stop_;
  std::mutex dual_mutex_;
  std::mutex end_mutex_;
  std::condition_variable cv_;
  std::vector<uint64_t> neighbor_ids_;
  std::unordered_map<uint64_t, MPI_Comm> communicator_map_;
  int* counter_storage_;
  MPI_Win counter_window_;

  // Store requests to allow for sending while performing other operations
  bool has_outgoing_data_;
  std::vector<MPI_Request> outgoing_comm_;
  std::unordered_map<uint64_t, std::vector<double>> outgoing_buffer_;

  int* flag_storage_;
  MPI_Win flag_window_;

  int* finish_flags_;
  MPI_Win finish_flag_window_;

  int iteration_counter_;
  unsigned long global_counter_;
};

}  // namespace dba
