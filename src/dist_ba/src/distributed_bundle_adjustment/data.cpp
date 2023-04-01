#include <gflags/gflags.h>
#include <glog/logging.h>
#include <experimental/filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>

#include "distributed_bundle_adjustment/data.hpp"

DEFINE_string(admm_type, "central_sync",
              "Which ADMM that will be used, available are central_sync, "
              "central_async and decentral_async");
DEFINE_string(data_folder, "",
              "The path where the input data structures are stored.");
DEFINE_string(
    result_folder, "",
    "Folder where the intermediate and final results should be written to.");
DECLARE_string(data_folder);

DECLARE_double(alpha_map_points);
DECLARE_double(alpha_intrinsics);
DECLARE_double(alpha_distortion);
DECLARE_double(alpha_rotation);
DECLARE_double(alpha_translation);

namespace filesystem = std::experimental::filesystem;

namespace dba {

Data::Data(const uint64_t graph_id)
    : graph_id_(graph_id),
      max_frame_id_(0u),
      num_frames_global_(0),
      num_map_points_global_(0),
      num_observations_global_(0) {}

Data::~Data() {}

auto Data::updateFromOther(const DataSharedPtr data_ptr) -> bool {
  for (auto [id, frame_ptr] : frames_) {
    auto other_frame_ptr = data_ptr->getFrame(id);
    if (other_frame_ptr == nullptr) {
      LOG(WARNING) << "Frame was not found in data which should be copied";
      return false;
    }
    frame_ptr->p_W_C_ = other_frame_ptr->p_W_C_;
    frame_ptr->q_W_C_ = other_frame_ptr->q_W_C_;
    frame_ptr->intrinsics_ = other_frame_ptr->intrinsics_;
    frame_ptr->dist_coeffs_ = other_frame_ptr->dist_coeffs_;
    frame_ptr->is_valid_ = other_frame_ptr->is_valid_;
  }

  for (auto [id, map_point_ptr] : map_points_) {
    auto other_map_point_ptr = data_ptr->getMapPoint(id);
    if (other_map_point_ptr == nullptr) {
      LOG(WARNING) << "Map Point was not found in data which should be copied";
      return false;
    }
    map_point_ptr->position_ = other_map_point_ptr->position_;
    map_point_ptr->is_valid_ = other_map_point_ptr->is_valid_;
  }

  return true;
}

auto Data::readDataFromFiles(const std::string& path,
                             const std::string& file_extension) -> bool {
  CHECK(readGlobalProperties());
  std::string sub_folder_data, sub_folder_obs;
  if (graph_id_ < std::numeric_limits<uint64_t>::max()) {
    sub_folder_data = path + "/Graph_" + std::to_string(graph_id_);
    sub_folder_obs = FLAGS_data_folder + "/Graph_" + std::to_string(graph_id_);
  } else {
    sub_folder_data = path + "/Full";
    sub_folder_obs = FLAGS_data_folder + "/Full";
  }
  if (!filesystem::is_directory(sub_folder_data)) {
    LOG(WARNING) << "Could not find data_folder: " << sub_folder_data;
    return false;
  }
  if (!this->readFrames(sub_folder_data, file_extension)) {
    LOG(WARNING) << "Failed to read the frames from " << sub_folder_data;
    return false;
  }
  if (!this->readMapPoints(sub_folder_data, file_extension)) {
    LOG(WARNING) << "Failed to read the map points from " << sub_folder_data;
    return false;
  }
  if (!this->readObservations(sub_folder_obs)) {
    LOG(WARNING) << "Failed to read the observations from " << sub_folder_obs;
    return false;
  }
  if (!this->readConnectionGraph(FLAGS_data_folder)) {
    LOG(WARNING) << "Failed to read the connection graph from "
                 << FLAGS_data_folder;
    return false;
  }

  // Initialize the consensus weights
  initializeDualWeights();

  // Update the neighbors
  neighbor_ids_.clear();
  if (!frames_to_comm_.empty() || !map_points_to_comm_.empty()) {
    std::unordered_set<uint64_t> tmp;
    for (const auto& [id, frames] : frames_to_comm_) {
      tmp.insert(id);
    }
    for (const auto& [id, map_points_] : map_points_to_comm_) {
      tmp.insert(id);
    }
    neighbor_ids_.reserve(tmp.size());
    for (const auto& v : tmp) {
      neighbor_ids_.push_back(v);
    }
    std::sort(neighbor_ids_.begin(), neighbor_ids_.end());
  }

  return true;
}

auto Data::getMapPointIds() const -> std::vector<uint64_t> {
  std::vector<uint64_t> result;
  result.reserve(map_points_.size());
  for (const auto& [id, ptr] : map_points_) result.push_back(id);
  return result;
}

auto Data::getFrameIds() const -> std::vector<uint64_t> {
  std::vector<uint64_t> result;
  result.reserve(frames_.size());
  for (const auto& [id, ptr] : frames_) result.push_back(id);
  return result;
}

auto Data::getFrameIds(const uint64_t graph_id) const -> std::vector<uint64_t> {
  if (!graph_to_frames_.count(graph_id)) return std::vector<uint64_t>();
  return graph_to_frames_.at(graph_id);
}

auto Data::getMapPoint(const uint64_t id) const -> MapPointSharedPtr {
  if (!map_points_.count(id)) return nullptr;
  return map_points_.at(id);
}

auto Data::getCommMapPoints(const uint64_t graph_id) const
    -> std::vector<uint64_t> {
  if (!map_points_to_comm_.count(graph_id)) return std::vector<uint64_t>();
  return map_points_to_comm_.at(graph_id);
}

auto Data::getFrame(const uint64_t id) const -> FrameSharedPtr {
  if (!frames_.count(id)) return nullptr;
  return frames_.at(id);
}

auto Data::getCommFrames(const uint64_t graph_id) const
    -> std::vector<uint64_t> {
  if (!frames_to_comm_.count(graph_id)) return std::vector<uint64_t>();
  return frames_to_comm_.at(graph_id);
}

auto Data::writeOutResult(const std::string& ap) -> bool {
  std::lock_guard<std::mutex> lock(write_mutex_);
  if (frames_.empty() || map_points_.empty()) return false;
  const std::string data_folder =
      FLAGS_result_folder + "/Graph_" + std::to_string(graph_id_);
  if (!filesystem::is_directory(data_folder)) {
    LOG(WARNING) << "Creates folder at " << data_folder;
    CHECK(filesystem::create_directory(data_folder));
  }
  const std::string frame_file = data_folder + "/frames_opt_" + ap + ".csv";
  std::ofstream file;
  file.open(frame_file);
  for (const auto [id, frame_ptr] : frames_) {
    if (!frame_ptr->is_valid_) continue;
    frame_ptr->p_W_C_[0] =
        (std::abs(frame_ptr->p_W_C_[0]) > 1e-14) ? frame_ptr->p_W_C_[0] : 0.0;
    frame_ptr->p_W_C_[1] =
        (std::abs(frame_ptr->p_W_C_[1]) > 1e-14) ? frame_ptr->p_W_C_[1] : 0.0;
    frame_ptr->p_W_C_[2] =
        (std::abs(frame_ptr->p_W_C_[2]) > 1e-14) ? frame_ptr->p_W_C_[2] : 0.0;
    frame_ptr->q_W_C_.coeffs()[0] =
        (std::abs(frame_ptr->q_W_C_.coeffs()[0]) > 1e-14)
            ? frame_ptr->q_W_C_.coeffs()[0]
            : 0.0;
    frame_ptr->q_W_C_.coeffs()[1] =
        (std::abs(frame_ptr->q_W_C_.coeffs()[1]) > 1e-14)
            ? frame_ptr->q_W_C_.coeffs()[1]
            : 0.0;
    frame_ptr->q_W_C_.coeffs()[2] =
        (std::abs(frame_ptr->q_W_C_.coeffs()[2]) > 1e-14)
            ? frame_ptr->q_W_C_.coeffs()[2]
            : 0.0;
    frame_ptr->q_W_C_.coeffs()[3] =
        (std::abs(frame_ptr->q_W_C_.coeffs()[3]) > 1e-14)
            ? frame_ptr->q_W_C_.coeffs()[3]
            : 0.0;
    frame_ptr->intrinsics_[0] = (std::abs(frame_ptr->intrinsics_[0]) > 1e-14)
                                    ? frame_ptr->intrinsics_[0]
                                    : 0.0;
    frame_ptr->dist_coeffs_[0] = (std::abs(frame_ptr->dist_coeffs_[0]) > 1e-14)
                                     ? frame_ptr->dist_coeffs_[0]
                                     : 0.0;
    frame_ptr->dist_coeffs_[1] = (std::abs(frame_ptr->dist_coeffs_[1]) > 1e-14)
                                     ? frame_ptr->dist_coeffs_[1]
                                     : 0.0;
    file << std::setprecision(10) << id << "," << frame_ptr->p_W_C_[0] << ","
         << frame_ptr->p_W_C_[1] << "," << frame_ptr->p_W_C_[2] << ","
         << frame_ptr->q_W_C_.x() << "," << frame_ptr->q_W_C_.y() << ","
         << frame_ptr->q_W_C_.z() << "," << frame_ptr->q_W_C_.w() << ",";
    for (size_t i = 0; i < 4; ++i) {
      if (i < kNumIntrinsicParams) {
        file << frame_ptr->intrinsics_[i] << ",";
      } else {
        file << "x,";
      }
    }
    for (size_t i = 0; i < 5; ++i) {
      if (i < kNumDistortionParams) {
        if (i < 4) {
          file << frame_ptr->dist_coeffs_[i] << ",";
        } else {
          file << frame_ptr->dist_coeffs_[i] << std::endl;
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
  const std::string map_point_file =
      data_folder + "/map_points_opt_" + ap + ".csv";
  file.open(map_point_file);
  for (const auto [id, map_point_ptr] : map_points_) {
    file << std::setprecision(12) << id << "," << map_point_ptr->position_[0]
         << "," << map_point_ptr->position_[1] << ","
         << map_point_ptr->position_[2] << std::endl;
  }
  file.close();
}

auto Data::readGlobalProperties() -> bool {
  const std::string basefolder = FLAGS_data_folder + "/Full/";
  const std::string frame_file = basefolder + "frames.csv";
  num_frames_global_ = getNumberOfLines(frame_file);
  const std::string map_point_file = basefolder + "map_points.csv";
  num_map_points_global_ = getNumberOfLines(map_point_file);
  const std::string observation_file = basefolder + "observations.csv";
  num_observations_global_ = getNumberOfLines(observation_file);
  return (num_frames_global_ > 0) && (num_map_points_global_ > 0) &&
         (num_observations_global_ > 0);
}

auto Data::getNumberOfLines(const std::string& filename) -> size_t {
  size_t num_lines = 0;
  if (!filesystem::exists(filename)) {
    LOG(WARNING) << filename << " does not exist";
    return num_lines;
  }
  std::ifstream file;
  file.open(filename);
  std::string dummy_line;
  while (std::getline(file, dummy_line)) {
    ++num_lines;
  }
  return num_lines;
}

auto Data::readFrames(const std::string& data_folder,
                      const std::string& appendix) -> bool {
  const std::string file_ending =
      (appendix.empty()) ? "/frames.csv" : "/frames" + appendix + ".csv";
  const std::string filename = data_folder + file_ending;
  if (!filesystem::exists(filename)) {
    LOG(WARNING) << filename << " does not exist";
    return false;
  }
  std::ifstream file;
  file.open(filename, std::ifstream::in);
  std::string line;
  while (std::getline(file, line)) {
    std::stringstream ss(line);
    std::array<std::string, 17> data;
    for (size_t i = 0; i < 17; ++i) {
      if (!std::getline(ss, data[i], ',')) {
        return false;
      }
    }

    const uint64_t id = std::stoull(data[0]);
    max_frame_id_ = std::max(id, max_frame_id_);
    const Eigen::Vector3d p_W_C(std::stod(data[1]), std::stod(data[2]),
                                std::stod(data[3]));
    Eigen::Quaterniond q_W_C(std::stod(data[7]), std::stod(data[4]),
                             std::stod(data[5]), std::stod(data[6]));
    q_W_C.normalize();
    // TODO handle general case of camera parameters
    Eigen::VectorXd intrinsics;
    intrinsics.resize(kNumIntrinsicParams);
    for (size_t i = 0; i < kNumIntrinsicParams; ++i) {
      intrinsics[i] = std::stod(data[8 + i]);
    }
    Eigen::VectorXd dist_coeffs;
    dist_coeffs.resize(kNumDistortionParams);
    for (size_t i = 0; i < kNumDistortionParams; ++i) {
      dist_coeffs[i] = std::stod(data[8 + 4 + i]);
    }
    Camera::Type cam_type;
    if (kNumIntrinsicParams == 1) {
      cam_type = Camera::Type::kPinholeSimple;
    } else {
      cam_type = Camera::Type::kPinhole;
    }
    Distortion::Type dist_type;
    switch (kNumDistortionParams) {
      case 2: {
        dist_type = Distortion::Type::kRadDist;
        break;
      }
      case 4: {
        dist_type = Distortion::Type::kEquiDist;
        break;
      }
      case 5: {
        dist_type = Distortion::Type::kRadTanDist;
        break;
      }
    }
    FrameSharedPtr frame_ptr = nullptr;
    frame_ptr = std::make_shared<Frame>(id, graph_id_, p_W_C, q_W_C, intrinsics,
                                        dist_coeffs, cam_type, dist_type);
    FrameDual dual_init(id);
    dual_init.fill(0.0);
    frame_ptr->setCentralDual(dual_init);
    this->frames_[id] = frame_ptr;
    frame_ptr->average_state_.setId(id);
    frame_ptr->average_state_.fill(0.0);
    double* position_ptr = frame_ptr->average_state_.getPosition();
    for (size_t i = 0; i < 3; ++i) position_ptr[i] = frame_ptr->p_W_C_[i];
    double* intrinsics_ptr = frame_ptr->average_state_.getIntrisincs();
    for (size_t i = 0; i < kNumIntrinsicParams; ++i)
      intrinsics_ptr[i] = frame_ptr->intrinsics_[i];
    double* distortion_ptr = frame_ptr->average_state_.getDistortion();
    for (size_t i = 0; i < kNumDistortionParams; ++i)
      distortion_ptr[i] = frame_ptr->dist_coeffs_[i];
  }
  file.close();
  if (this->frames_.empty()) return false;
  return true;
}

auto Data::readMapPoints(const std::string& data_folder,
                         const std::string& appendix) -> bool {
  const std::string file_ending = (appendix.empty())
                                      ? "/map_points.csv"
                                      : "/map_points" + appendix + ".csv";
  const std::string filename = data_folder + file_ending;
  if (!filesystem::exists(filename)) return false;
  std::ifstream file;
  file.open(filename, std::ifstream::in);
  std::string line;
  while (std::getline(file, line)) {
    std::stringstream ss(line);
    std::array<std::string, 4> data;
    for (size_t i = 0; i < 4; ++i) {
      if (!std::getline(ss, data[i], ',')) return false;
    }
    const uint64_t id = std::stoull(data[0]);
    const Eigen::Vector3d p_W(std::stod(data[1]), std::stod(data[2]),
                              std::stod(data[3]));
    MapPointSharedPtr map_point_ptr =
        std::make_shared<MapPoint>(id, graph_id_, p_W);
    this->map_points_[id] = map_point_ptr;
  }
  file.close();
  if (this->map_points_.empty()) return false;
  return true;
}

auto Data::readObservations(const std::string& data_folder) -> bool {
  const std::string filename = data_folder + "/observations.csv";
  if (!filesystem::exists(filename)) return false;
  std::ifstream file;
  file.open(filename, std::ifstream::in);
  std::string line;
  while (std::getline(file, line)) {
    std::stringstream ss(line);
    std::array<std::string, 4> data;
    for (size_t i = 0; i < 4; ++i) {
      if (!std::getline(ss, data[i], ',')) return false;
    }
    Observation obs;
    obs.frame_id = std::stoull(data[0]);
    obs.mp_id = std::stoull(data[1]);
    obs.obs[0] = std::stod(data[2]);
    obs.obs[1] = std::stod(data[3]);
    if (!frames_.count(obs.frame_id)) {
      continue;
      std::cout << "cannot find frame " << obs.frame_id << std::endl;
      return false;
    }
    if (!frames_[obs.frame_id]->addObservation(obs)) return false;
    if (!map_points_.count(obs.mp_id)) {
      std::cout << "cannot find map point " << obs.mp_id << std::endl;
      return false;
    }
    map_points_[obs.mp_id]->addObserverId(obs.frame_id);
  }
  file.close();
  return true;
}

auto Data::readConnectionGraph(const std::string& data_folder) -> bool {
  const std::string filename = data_folder + "/connections.csv";
  if (!filesystem::exists(filename)) return false;
  std::ifstream file;
  file.open(filename, std::ifstream::in);
  std::string line;
  while (std::getline(file, line)) {
    std::stringstream ss(line);
    std::string ids, num_conns;
    if (!std::getline(ss, ids, ',')) return false;
    if (!std::getline(ss, num_conns, ',')) return false;
    const size_t num_conn = static_cast<size_t>(std::stoi(num_conns));
    const uint64_t id = std::stoull(ids);

    if (id <= max_frame_id_) {
      // This corresponds to a camera consensus
      if (!frames_.count(id)) continue;
      auto frame_ptr = frames_[id];
      CHECK(frame_ptr != nullptr);
      for (size_t i = 0; i < num_conn; ++i) {
        std::string tmp;
        if (!std::getline(ss, tmp, ',')) return false;
        uint64_t neigh_id = std::stoull(tmp);
        graph_to_frames_[neigh_id].push_back(id);
        if (num_conn >= 2) {
          frames_to_comm_[neigh_id].push_back(id);
        }
      }
    } else {
      // This corresponds to a point consensus
      if (!map_points_.count(id)) continue;
      std::string tmp;
      auto map_point_ptr = map_points_[id];
      const double lambda_map_point = map_point_ptr->lambda_;
      for (size_t i = 0; i < num_conn; ++i) {
        if (!std::getline(ss, tmp, ',')) return false;
        uint64_t neigh_id = std::stoull(tmp);
        graph_to_map_points_[neigh_id].push_back(id);
        if (num_conn >= 2) {
          map_points_to_comm_[neigh_id].push_back(id);
        }
      }
    }
  }
  file.close();
  return true;
}

auto Data::initializeDualWeights() -> bool {
  // Set the base lambdas
  const double lambda_translation_base =
      FLAGS_alpha_translation * num_observations_global_ /
      static_cast<double>(num_frames_global_);
  const double lambda_rotation_base = FLAGS_alpha_rotation *
                                      num_observations_global_ /
                                      static_cast<double>(num_frames_global_);
  const double lambda_intrinsics_base = FLAGS_alpha_intrinsics *
                                        num_observations_global_ /
                                        static_cast<double>(num_frames_global_);
  const double lambda_distortion_base = FLAGS_alpha_distortion *
                                        num_observations_global_ /
                                        static_cast<double>(num_frames_global_);
  const double lambda_map_point_base =
      FLAGS_alpha_map_points * num_observations_global_ /
      static_cast<double>(num_map_points_global_);

  for (auto [id, frame_ptr] : frames_) {
    // Initialize the consensus weights for the centralized approach
    frame_ptr->sigma_trans_ = lambda_translation_base;
    frame_ptr->sigma_rot_ = lambda_rotation_base;
    frame_ptr->sigma_intr_ = lambda_intrinsics_base;
    frame_ptr->sigma_dist_ = lambda_distortion_base;

    // Initialize the dual weights for the asynchronous approach
    std::unordered_set<uint64_t> tmp_neigh_ids;
    tmp_neigh_ids.insert(graph_id_);
    for (const auto& [n_id, frame_ids] : frames_to_comm_) {
      if (std::find(frame_ids.begin(), frame_ids.end(), id) !=
          frame_ids.end()) {
        tmp_neigh_ids.insert(n_id);
      }
    }
    std::vector<uint64_t> neigh_ids(tmp_neigh_ids.begin(), tmp_neigh_ids.end());
    const size_t num_neighs = neigh_ids.size();
    frame_ptr->lambda_trans_ =
        lambda_translation_base;  // / static_cast<double>(num_neighs);
    frame_ptr->lambda_rot_ =
        lambda_rotation_base;  // / static_cast<double>(num_neighs);
    frame_ptr->lambda_intr_ =
        lambda_intrinsics_base;  // / static_cast<double>(num_neighs);
    frame_ptr->lambda_dist_ =
        lambda_distortion_base;  // / static_cast<double>(num_neighs);

    if (FLAGS_admm_type == "decentral_async") {
      for (const auto& n_id : neigh_ids) {
        FrameDual dual(id);
        double* position_ptr = dual.getPosition();
        for (size_t j = 0; j < 3; ++j) {
          position_ptr[j] = -frame_ptr->lambda_trans_ * frame_ptr->p_W_C_[j];
        }
        double* intrinsics_ptr = dual.getIntrisincs();
        for (size_t j = 0; j < kNumIntrinsicParams; ++j) {
          intrinsics_ptr[j] =
              -frame_ptr->lambda_intr_ * frame_ptr->intrinsics_[j];
        }
        double* distortion_ptr = dual.getDistortion();
        for (size_t j = 0; j < kNumDistortionParams; ++j) {
          distortion_ptr[j] =
              -frame_ptr->lambda_dist_ * frame_ptr->dist_coeffs_[j];
        }
        frame_ptr->setDualData(n_id, dual);
        frame_ptr->setCommDual(n_id, dual);
      }
    }
  }

  for (auto [id, map_point_ptr] : map_points_) {
    // Initialize the consensus weights for the centralized approach
    map_point_ptr->sigma_ = lambda_map_point_base;

    // Initialize the dual weights for the asynchronous approach
    std::unordered_set<uint64_t> tmp_neigh_ids;
    tmp_neigh_ids.insert(graph_id_);
    for (const auto& [n_id, map_point_ids] : map_points_to_comm_) {
      if (std::find(map_point_ids.begin(), map_point_ids.end(), id) !=
          map_point_ids.end()) {
        tmp_neigh_ids.insert(n_id);
      }
    }
    std::vector<uint64_t> neigh_ids(tmp_neigh_ids.begin(), tmp_neigh_ids.end());
    const size_t num_neighs = neigh_ids.size();
    map_point_ptr->lambda_ =
        lambda_map_point_base;  // / static_cast<double>(num_neighs);
    if (FLAGS_admm_type == "decentral_async") {
      for (const auto& n_id : neigh_ids) {
        MapPointDual dual(id);
        double* position_ptr = dual.getPosition();
        for (size_t j = 0; j < 3; ++j) {
          position_ptr[j] =
              -map_point_ptr->lambda_ * map_point_ptr->position_[j];
        }
        map_point_ptr->setDualData(n_id, dual);
        map_point_ptr->setCommDual(n_id, dual);
      }
    }
  }
  return true;
}

}  // namespace dba
