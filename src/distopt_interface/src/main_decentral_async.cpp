#include <gflags/gflags.h>
#include <glog/logging.h>
#include <chrono>
#include <fstream>
#include <iostream>

#include "rclcpp/rclcpp.hpp"

// uncomment once we've written our async comm file
#include "distributed_bundle_adjustment/asynchronous_communication.hpp"
#include "distributed_bundle_adjustment/common.hpp"
#include "distributed_bundle_adjustment/data.hpp"
#include "distributed_bundle_adjustment/distortion.hpp"
#include "distributed_bundle_adjustment/optimization.hpp"

DECLARE_string(admm_type);
DECLARE_string(data_folder);
DECLARE_string(result_folder);
DECLARE_uint64(num_admm_iter);

/**
 * Must modify this thread to run as a ROS node
*/
int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  CHECK(FLAGS_admm_type == "decentral_async");

  rclcpp::init(argc, argv);
  rclcpp::NodeOptions options;
  options.use_intra_process_comms(true);
  rclcpp::Node::SharedPtr node = std::make_shared<rclcpp::Node>("my_node", "", options);

  int world_size = 2;
  int rank;
  if (node->has_parameter("rank")) {
    node->get_parameter("rank", rank);
  } else {
    rank = 0;
  }

  if (rank == 0) {
    // This is the coordinator and storage node
    const size_t num_nodes = world_size - 1;
    std::vector<dba::DataSharedPtr> data(num_nodes);
    for (size_t i = 0; i < num_nodes; ++i) {
      data[i] = std::make_shared<dba::Data>(i);
      CHECK(data[i]->readDataFromFiles(FLAGS_data_folder));
    }
    dba::AsynchronousCoordinator coordinator(data);
  } else {
    // These are the actual worker nodes
    dba::DataSharedPtr data_ptr(new dba::Data(rank - 1));
    dba::DataSharedPtr data_copy_ptr(new dba::Data(rank - 1));
    CHECK(data_ptr->readDataFromFiles(FLAGS_data_folder));
    CHECK(data_copy_ptr->readDataFromFiles(FLAGS_data_folder));
    dba::AsynchronousCommunication node(data_ptr, data_copy_ptr);
  }

  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
