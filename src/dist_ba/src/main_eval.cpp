#include <gflags/gflags.h>
#include <glog/logging.h>
#include <experimental/filesystem>
#include <fstream>
#include <iomanip>

#include "distributed_bundle_adjustment/common.hpp"
#include "distributed_bundle_adjustment/data.hpp"
#include "distributed_bundle_adjustment/optimization.hpp"
#include "distributed_bundle_adjustment/utils.hpp"

DEFINE_uint32(num_subgraphs, 0, "The number of subgraphs.");
DEFINE_uint32(max_iter_eval, 0, "The umber of evaluation iterations.");
DECLARE_uint32(write_out_iter);
DECLARE_string(result_folder);
DECLARE_string(data_folder);
DECLARE_string(admm_type);

namespace filesystem = std::experimental::filesystem;

auto averageResult(const std::vector<dba::DataSharedPtr>& data)
    -> std::vector<double> {
  using FrameDual = dba::PoseDual<kNumIntrinsicParams, kNumDistortionParams>;

  // Summarize the relevant data from all subgraphs.
  // Note: Currently we assume pose consensus, meaning that the map points are
  // unique.
  std::unordered_map<uint64_t, std::vector<FrameDual>> frame_data;
  std::unordered_map<uint64_t, Eigen::Quaterniond> ref_rot;
  std::unordered_map<uint64_t, dba::VectorOfVector3> map_point_data;
  for (const auto& data_ptr : data) {
    auto frame_ids = data_ptr->getFrameIds();
    for (const auto& id : frame_ids) {
      auto frame_ptr = data_ptr->getFrame(id);
      if (ref_rot.find(id) == ref_rot.end()) {
        ref_rot[id] = frame_ptr->getReferenceRotation();
      }
      CHECK(frame_ptr != nullptr);
      FrameDual tmp(id);
      Eigen::Vector3d delta_q;
      dba::utils::rotmath::Minus(frame_ptr->q_W_C_,
                                 frame_ptr->getReferenceRotation(), &delta_q);
      for (size_t i = 0; i < tmp.getSize(); ++i) {
        if (i < 3) {
          // Translation
          tmp[i] = frame_ptr->p_W_C_[i];
        } else if (i >= 3 && i < 6) {
          // Rotation
          tmp[i] = delta_q[i - 3];
        } else if (i >= 6 && i < 6 + kNumIntrinsicParams) {
          // Intrinsics
          tmp[i] = frame_ptr->intrinsics_[i - 6];
        } else {
          // Distortion
          tmp[i] = frame_ptr->dist_coeffs_[i - 6 - kNumIntrinsicParams];
        }
      }
      frame_data[id].push_back(tmp);
    }
    const auto mp_ids = data_ptr->getMapPointIds();
    for (const auto& id : mp_ids) {
      auto mp_ptr = data_ptr->getMapPoint(id);
      CHECK(mp_ptr != nullptr);
      map_point_data[id].push_back(mp_ptr->position_);
    }
  }
  std::vector<uint64_t> all_frame_ids;
  for (const auto& [id, data] : frame_data) all_frame_ids.push_back(id);
  std::sort(all_frame_ids.begin(), all_frame_ids.end());

  // Perform the actual averaging and write out the result
  if (!filesystem::is_directory(FLAGS_result_folder + "/Full")) {
    LOG(WARNING) << "Creates folder at " << FLAGS_result_folder << "/Full";
    CHECK(filesystem::create_directory(FLAGS_result_folder + "/Full"));
  }
  std::string filename_frames = FLAGS_result_folder + "/Full/frames_avg_";
  std::string filename_mps = FLAGS_result_folder + "/Full/map_points_avg_";
  std::string appendix = "_avg_";
  if (FLAGS_admm_type == "central_sync") {
    filename_frames += "cent_sync.csv";
    filename_mps += "cent_sync.csv";
    appendix += "cent_sync";
  } else if (FLAGS_admm_type == "central_async") {
    filename_frames += "cent_async.csv";
    filename_mps += "cent_async.csv";
    appendix += "cent_async";
  } else if (FLAGS_admm_type == "decentral_async") {
    filename_frames += "decent_async.csv";
    filename_mps += "decent_async.csv";
    appendix += "decent_async";
  } else {
    LOG(FATAL) << "Unknown parameter for admm_type (" << FLAGS_admm_type << ")";
  }

  std::ofstream file;
  file.open(filename_frames);
  for (const auto& f_id : all_frame_ids) {
    const auto& tmp_data = frame_data.at(f_id);
    Eigen::Vector3d delta_q_avg = Eigen::Vector3d::Zero();
    Eigen::Vector3d p_W_C_avg = Eigen::Vector3d::Zero();
    Eigen::Matrix<double, kNumIntrinsicParams, 1> intrinsics_avg;
    intrinsics_avg.setZero();
    Eigen::Matrix<double, kNumDistortionParams, 1> distortion_avg;
    distortion_avg.setZero();
    for (const auto& d : tmp_data) {
      for (size_t i = 0; i < d.getSize(); ++i) {
        if (i < 3) {
          p_W_C_avg[i] += d[i];
        } else if (i >= 3 && i < 6) {
          delta_q_avg[i - 3] += d[i];
        } else if (i >= 6 && i < 6 + kNumIntrinsicParams) {
          intrinsics_avg[i - 6] += d[i];
        } else {
          distortion_avg[i - 6 - kNumIntrinsicParams] += d[i];
        }
      }
    }
    const double N = static_cast<double>(tmp_data.size());
    p_W_C_avg /= N;
    delta_q_avg /= N;
    intrinsics_avg /= N;
    distortion_avg /= N;

    Eigen::Quaterniond q_W_C_avg;
    Eigen::Quaterniond q_ref = ref_rot[f_id];
    dba::utils::rotmath::Plus(q_ref, delta_q_avg, &q_W_C_avg);
    file << std::setprecision(12) << f_id << "," << p_W_C_avg[0] << ","
         << p_W_C_avg[1] << "," << p_W_C_avg[2] << "," << q_W_C_avg.x() << ","
         << q_W_C_avg.y() << "," << q_W_C_avg.z() << "," << q_W_C_avg.w()
         << ",";
    for (size_t i = 0; i < 4; ++i) {
      if (i < kNumIntrinsicParams) {
        file << intrinsics_avg[i] << ",";
      } else {
        file << "x,";
      }
    }
    for (size_t i = 0; i < 5; ++i) {
      if (i < kNumDistortionParams) {
        if (i < 4) {
          file << distortion_avg[i] << ",";
        } else {
          file << distortion_avg[i] << std::endl;
        }
      } else {
        if (i < 4) {
          file << "x,";
        } else {
          file << "x" << std::endl;
        }
      }
    }
  }
  file.close();

  file.open(filename_mps);
  for (const auto& [id, data] : map_point_data) {
    Eigen::Vector3d position = Eigen::Vector3d::Zero();
    for (const auto& d : data) {
      position[0] += d[0];
      position[1] += d[1];
      position[2] += d[2];
    }
    const int N = data.size();
    position /= N;
    file << std::setprecision(12) << id << "," << position[0] << ","
         << position[1] << "," << position[2] << std::endl;
  }
  file.close();

  dba::DataSharedPtr comb_data_ptr =
      std::make_shared<dba::Data>(std::numeric_limits<uint64_t>::max());

  CHECK(comb_data_ptr->readDataFromFiles(FLAGS_result_folder, appendix));
  dba::Optimization optimizer(comb_data_ptr,
                              dba::Optimization::ConsensusType::kNoConsensus);
  CHECK(optimizer.setupProblem());
  return optimizer.computeErrors();
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  CHECK(FLAGS_admm_type == "central_sync" ||
        FLAGS_admm_type == "central_async" ||
        FLAGS_admm_type == "decentral_async");
  std::string filename = FLAGS_result_folder;
  std::string pre_app;
  if (FLAGS_admm_type == "central_sync") {
    filename += "/errors_cent_sync.csv";
    pre_app = "_opt_cent_sync_";
  } else if (FLAGS_admm_type == "central_async") {
    filename += "/errors_cent_async.csv";
    pre_app = "_opt_cent_async_";
  } else if (FLAGS_admm_type == "decentral_async") {
    filename += "/errors_decent_async.csv";
    pre_app = "_opt_decent_async_";
  } else {
    LOG(FATAL) << "Unknown parameter for admm_type (" << FLAGS_admm_type << ")";
  }

  std::vector<std::vector<double>> data;
  for (int j = 0; j <= FLAGS_max_iter_eval; j += FLAGS_write_out_iter) {
    std::vector<dba::DataSharedPtr> eval_data(FLAGS_num_subgraphs, nullptr);
    for (int i = 0; i < FLAGS_num_subgraphs; ++i) {
      eval_data[i] = std::make_shared<dba::Data>(i);
      std::string app = pre_app + std::to_string(j);
      if (j == 0)
        eval_data[i]->readDataFromFiles(FLAGS_data_folder);
      else
        eval_data[i]->readDataFromFiles(FLAGS_result_folder, app);
    }
    data.push_back(averageResult(eval_data));
  }
  std::ofstream file;
  file.open(filename);
  for (int i = 0; i < data[0].size(); ++i) {
    for (int j = 0; j < data.size(); ++j) {
      file << data[j][i];
      if (j + 1 != data.size()) file << ",";
    }
    file << std::endl;
  }
  file.close();
  return 0;
}
