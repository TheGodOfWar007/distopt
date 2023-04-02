#include <gflags/gflags.h>
#include <glog/logging.h>
#include <chrono>
#include <iostream>

#include <mpi.h>

#include "distributed_bundle_adjustment/central_async_communication.hpp"
#include "distributed_bundle_adjustment/common.hpp"
#include "distributed_bundle_adjustment/data.hpp"
#include "distributed_bundle_adjustment/distortion.hpp"
#include "distributed_bundle_adjustment/optimization.hpp"

DECLARE_string(admm_type);
DECLARE_string(data_folder);

using FrameDual = dba::PoseDual<kNumIntrinsicParams, kNumDistortionParams>;

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  CHECK(FLAGS_admm_type == "central_async");
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

    // Initialize the dual variables
    for (uint64_t i = 0; i < static_cast<uint64_t>(world_size - 1); ++i) {
      // For the frames
      const auto& frame_ids = data_ptr->getCommFrames(i);
      for (const auto& id : frame_ids) {
        auto frame_ptr = data_ptr->getFrame(id);
        CHECK(frame_ptr != nullptr);
        FrameDual initial_dual(id);
        initial_dual.fill(0.0);
        frame_ptr->setDualData(i, initial_dual);
      }
      // For the map points
      const auto& map_point_ids = data_ptr->getCommMapPoints(i);
      for (const auto& id : map_point_ids) {
        auto map_point_ptr = data_ptr->getMapPoint(id);
        CHECK(map_point_ptr != nullptr);
        dba::MapPointDual initial_dual(id);
        initial_dual.fill(0.0);
        map_point_ptr->setDualData(i, initial_dual);
      }
    }
    dba::AsyncMasterNode master(world_size - 1, data_ptr);
    master.startNodes();
  } else {
    data_ptr = std::make_shared<dba::Data>(rank - 1);
    CHECK(data_ptr->readDataFromFiles(FLAGS_data_folder));
    dba::AsyncWorkerNode worker(world_size - 1, rank, data_ptr);
  }
  MPI_Finalize();
  return 0;
}
