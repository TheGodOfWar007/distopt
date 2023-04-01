#include <gflags/gflags.h>
#include <glog/logging.h>
#include <chrono>
#include <iostream>

#include <mpi.h>

#include "distributed_bundle_adjustment/central_communication.hpp"
#include "distributed_bundle_adjustment/data.hpp"
#include "distributed_bundle_adjustment/distortion.hpp"
#include "distributed_bundle_adjustment/optimization.hpp"

DECLARE_string(admm_type);
DECLARE_string(data_folder);

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  CHECK(FLAGS_admm_type == "central_sync");
  int prov;
  auto stat = MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &prov);
  CHECK(stat == MPI_SUCCESS);
  CHECK(prov == MPI_THREAD_MULTIPLE);
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  dba::DataSharedPtr data_ptr;
  if (rank == 0) {
    data_ptr =
        std::make_shared<dba::Data>(std::numeric_limits<uint64_t>::max());
    CHECK(data_ptr->readDataFromFiles(FLAGS_data_folder));
    dba::MasterNode master(world_size - 1, data_ptr);
    std::cout << "should start" << std::endl;
    master.startNodes();
  } else {
    data_ptr = std::make_shared<dba::Data>(rank - 1);
    CHECK(data_ptr->readDataFromFiles(FLAGS_data_folder));
    std::cout << "Worker " << rank << " should start" << std::endl;
    dba::WorkerNode worker(rank, data_ptr);
  }
  MPI_Finalize();
  return 0;
}
