#include <condition_variable>
#include <memory>
#include <mutex>
#include <stack>
#include <thread>
#include <vector>

#include "distributed_bundle_adjustment/common.hpp"
#include "distributed_bundle_adjustment/data.hpp"
#include "distributed_bundle_adjustment/optimization.hpp"

#include "rclcpp/rclcpp.hpp"
#include "msg/node_data.hpp"
#include "msg/update.hpp"
#include "msg/finish_flag.hpp"
#include "msg/flag.hpp"
#include "msg/counter.hpp"
#include "msg/update.hpp"

/// @brief writing to dba namespace
namespace dba {

/// @brief AsynchronousCoordinator class definition
/// @details coordinates distributed optimization process
class AsynchronousCoordinator {
 public:
  /// @brief PoseDual type alias with kNumIntrinsicParams and kNumDistortionParams
  using FrameDual = PoseDual<kNumIntrinsicParams, kNumDistortionParams>;
  /// @brief type alias for shared pointer to a Data object
  using DataSharedPtrVector = std::vector<DataSharedPtr>;
  /// @brief map of DataIds to DataSharedPtrs
  using DataIdMap = std::unordered_map<uint64_t, DataSharedPtr>;
  /// @brief type alias for pair of two DataIds
  using EdgeId = std::pair<uint64_t, uint64_t>;
  /// @brief map of EdgeIds to some DataType using pair_hash hashing function
  /// @tparam DataType 
  template <class DataType>
  using EdgeToDataMap = std::unordered_map<EdgeId, DataType, pair_hash>;

 public:
  /// @brief delete default AsynchronousCoordinator constructor
  AsynchronousCoordinator() = delete;
  /// @brief AsynchronousCoordinator constructor declaration
  /// @param data data input
  AsynchronousCoordinator(const DataSharedPtrVector& data, const std::string& node_name);
  /// @brief AsynchronousCoordinator class destroyer
  ~AsynchronousCoordinator();

 private:
  /// @brief initialize rosmsgs windows for the buffers used by the Async Coordinator
  /// @return
  auto initializeBuffers() -> void;

  /// @brief main loop for AsynchronousCoorindator process
  /// @details run the coordinated distributed optimization process in a separate thread and update counter and buffers and send updates to appropriate nodes
  /// @return 
  auto mainThread() -> void;

  /// @brief update iteration counter used to track latest iteration of all nodes
  /// @return boolean of whether the counter was updated or not
  auto updateCounter() -> bool;

  /// @brief check graph edges if updates are required
  /// @return 
  auto updateBuffers() -> void;

  /// @brief check if the input buffer with a desired Edge needs to be updated and update if needed
  /// @param id EdgeId
  /// @param buffer 
  /// @return 
  auto checkAndUpdateBuffer(const EdgeId& id, std::vector<double>& buffer)
      -> bool;

  /// @brief send updates to neighbors of Node with node_id
  /// @param node_id
  /// @return 
  auto sendUpdates(const uint64_t& node_id) -> void;

  // Useful handles
  int world_size_;

  /// @brief DataIdMap mapping DataIds to DataSharedPtrs
  DataIdMap data_map_;

  // Storage for maintaining the iteration counts
  int* counter_storage_;
  rclcpp::Publisher<distopt_interface::msg::Counter>::SharedPtr counter_publisher_;
  int iteration_counter_;

  // Keep a flag array to check whether a node has finished
  int* finish_flags_;
  rclcpp::Publisher<distopt_interface::msg::FinishFlag>::SharedPtr finish_flag_publisher_;

  int* flag_storage_;
  rclcpp::Publisher<distopt_interface::msg::Flag>::SharedPtr flag_publisher_;
  std::vector<int> last_flag_state_;
  EdgeToDataMap<std::vector<double>> data_buffer_map_;
  // EdgeToDataMap<MPI_Comm> communicator_map_; // no clue how tf to change this lol
};

class AsynchronousCommunication {
 public:
  /// @brief PoseDual type alias with kNumIntrinsicParams and kNumDistortionParams
  using FrameDual = PoseDual<kNumIntrinsicParams, kNumDistortionParams>;

 public:
  /// @brief delete default AsynchronousCoordinator constructor
  AsynchronousCommunication() = delete;
  /// @brief AsynchronousCoordinator constructor declaration
  /// @param data data input
  AsynchronousCommunication(DataSharedPtr data_ptr,
                            DataSharedPtr data_copy_ptr);
  /// @brief AsynchronousCoordinator class destroyer
  ~AsynchronousCommunication();

 private:
  /// @brief main loop of AsynchronousCommunication process
  /// @return 
  auto mainThread() -> void;

  /// @brief perform Data communication between Nodes
  /// @details send Data to all neighbors by copying it to a local buffer
  /// @return 
  auto communicateData() -> void;

  /// @brief update Node using recevied Duals from neighboring Nodes
  /// @details signal that it is ready to receive updates, receive updated
  /// duals for each neighboring Node for Frames and MapPoints, and update
  /// corresponding objects
  /// @return 
  auto updateReceivedDuals() -> void;

  /// @brief update Duals for all Frames and MapPoints
  /// @details iterate over all Frames and MapPoints, call updateDualVariables
  /// for each, updating all dual variables
  /// @return 
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
  std::unordered_map<uint64_t, rclcpp::Publisher<distopt_interface::msg::NodeData>> publisher_map_;
  int* counter_storage_;
  rclcpp::Publisher<distopt_interface::msg::Counter>::SharedPtr counter_pub_;

  // Store requests to allow for sending while performing other operations
  bool has_outgoing_data_;
  std::vector<rclcpp::Client<srv::DataTransfer>::SharedPtr> outgoing_comm_;
  std::unordered_map<uint64_t, std::vector<double>> outgoing_buffer_;

  // Keep a flag array to check whether a node has finished
  int* finish_flags_;
  rclcpp::Publisher<distopt_interface::msg::FinishFlag>::SharedPtr finish_flag_publisher_;

  int* flag_storage_;
  rclcpp::Publisher<distopt_interface::msg::Flag>::SharedPtr flag_publisher_;

  int iteration_counter_;
  unsigned long global_counter_;
};

}  // namespace dba
