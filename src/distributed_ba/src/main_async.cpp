#include <gflags/gflags.h>
#include <glog/logging.h>
#include <chrono>
#include <fstream>
#include <iostream>

#include <mpi.h>

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
  int prov;
  // Replace MPI w/ ROS
  auto stat = MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &prov);
  CHECK(stat == MPI_SUCCESS);
  CHECK(prov == MPI_THREAD_SERIALIZED);
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  CHECK_GT(world_size, 1);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // What's the difference between these two??
  if (rank == 0) {
    // This is the coordinator and storage node
    const size_t num_nodes = world_size - 1;
    std::vector<dba::DataSharedPtr> data(num_nodes);
    for (size_t i = 0; i < num_nodes; ++i) {
      data[i] = std::make_shared<dba::Data>(i);
      CHECK(data[i]->readDataFromFiles(FLAGS_data_folder));
    }
    MPI_Barrier(MPI_COMM_WORLD);
    dba::AsynchronousCoordinator coordinator(data);
  } else {
    // These are the actual worker nodes
    dba::DataSharedPtr data_ptr(new dba::Data(rank - 1));
    dba::DataSharedPtr data_copy_ptr(new dba::Data(rank - 1));
    CHECK(data_ptr->readDataFromFiles(FLAGS_data_folder));
    CHECK(data_copy_ptr->readDataFromFiles(FLAGS_data_folder));
    MPI_Barrier(MPI_COMM_WORLD);
    dba::AsynchronousCommunication node(data_ptr, data_copy_ptr);
  }

  MPI_Finalize();
  return 0;
}
