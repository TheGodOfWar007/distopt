#include <gflags/gflags.h>
#include <glog/logging.h>
#include <chrono>
#include <experimental/filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>

#include "distributed_bundle_adjustment/common.hpp"
#include "distributed_bundle_adjustment/data.hpp"
#include "distributed_bundle_adjustment/optimization.hpp"

DECLARE_string(data_folder);
DECLARE_string(result_folder);

namespace filesystem = std::experimental::filesystem;

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  dba::DataSharedPtr data_ptr;
  data_ptr = std::make_shared<dba::Data>(std::numeric_limits<uint64_t>::max());
  if (data_ptr->readDataFromFiles(FLAGS_data_folder)) {
    LOG(INFO) << "Successfully read the data";
    dba::Optimization optimizer(data_ptr,
                                dba::Optimization::ConsensusType::kNoConsensus);
    optimizer.setupProblem();
    optimizer.performOptimization();
    const auto& errors = optimizer.computeErrors();
    const std::string data_folder = FLAGS_result_folder + "/Full";
    if (!filesystem::is_directory(data_folder)) {
      CHECK(filesystem::create_directory(data_folder));
    }
    std::ofstream file(data_folder + "/residuals_cent.csv");
    for (const auto e : errors) {
      file << e << std::endl;
    }
    file.close();

    const auto& frame_ids = data_ptr->getFrameIds();
    file.open(FLAGS_result_folder + "/Full/frames_opt_cent.csv");
    for (const auto id : frame_ids) {
      auto frame_ptr = data_ptr->getFrame(id);
      CHECK(frame_ptr != nullptr);
      file << std::setprecision(15) << id << "," << frame_ptr->p_W_C_[0] << ","
           << frame_ptr->p_W_C_[1] << "," << frame_ptr->p_W_C_[2] << ","
           << frame_ptr->q_W_C_.x() << "," << frame_ptr->q_W_C_.y() << ","
           << frame_ptr->q_W_C_.z() << "," << frame_ptr->q_W_C_.w() << ",";
      for (size_t i = 0; i < 4; ++i) {
        if (i < kNumIntrinsicParams) {
          file << std::setprecision(15) << frame_ptr->intrinsics_[i] << ",";
        } else {
          file << "x,";
        }
      }
      for (size_t i = 0; i < 5; ++i) {
        if (i < kNumDistortionParams) {
          file << std::setprecision(15) << frame_ptr->dist_coeffs_[i];
        } else {
          file << "x";
        }
        if (i < 4) {
          file << ",";
        }
      }
      file << std::endl;
    }
    file.close();

    const auto& mp_ids = data_ptr->getMapPointIds();
    file.open(FLAGS_result_folder + "/Full/map_points_opt_cent.csv");
    for (const auto id : mp_ids) {
      auto mp_ptr = data_ptr->getMapPoint(id);
      CHECK(mp_ptr != nullptr);
      file << std::setprecision(15) << id << "," << mp_ptr->position_[0] << ","
           << mp_ptr->position_[1] << "," << mp_ptr->position_[2] << std::endl;
    }
    file.close();

  } else {
    LOG(WARNING) << "Failed to read the data";
  }
  return 0;
}
