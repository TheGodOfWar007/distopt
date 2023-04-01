#include <gflags/gflags.h>
#include <glog/logging.h>
#include <chrono>
#include <experimental/filesystem>
#include <fstream>
#include <iomanip>

#include <mpi.h>

#include "distributed_bundle_adjustment/data.hpp"
#include "distributed_bundle_adjustment/optimization.hpp"
#include "distributed_bundle_adjustment/utils.hpp"

DEFINE_uint32(num_subgraphs, 0, "The number of subgraphs.");
DEFINE_uint32(max_iter_eval, 0, "The umber of evaluation iterations.");
DECLARE_uint32(write_out_iter);
DECLARE_string(result_folder);
DECLARE_string(data_folder);
DECLARE_string(admm_type);

constexpr double kEps = 1e-9;
constexpr size_t num_frame_vars =
    1 + 6 + kNumIntrinsicParams + kNumDistortionParams;
using FrameData = dba::PoseDual<kNumIntrinsicParams, kNumDistortionParams>;

auto computeMetrics(const std::vector<double>& errors)
    -> std::tuple<double, double, double> {
  std::vector<double> normal_errors(errors.size() / 2);
  const size_t max_normal_err = normal_errors.size();
  double mean = 0.0;
  size_t mean_count = 0;
  for (size_t i = 0, j = 0; i < errors.size(); i += 2) {
    CHECK_LT(j, max_normal_err);
    normal_errors[j] =
        std::sqrt(errors[i] * errors[i] + errors[i + 1] * errors[i + 1]);
    if (normal_errors[j] < 30.0) {
      mean += normal_errors[j];
      ++mean_count;
    }
    ++j;
  }
  mean = mean / mean_count;
  double standard_deviation = 0.0;
  mean_count = 0;
  for (const auto& err : normal_errors) {
    if (err < 30.0) {
      standard_deviation += (err - mean) * (err - mean);
      ++mean_count;
    }
  }
  standard_deviation = standard_deviation / mean_count;
  const size_t n = normal_errors.size() / 2;
  std::nth_element(normal_errors.begin(), normal_errors.begin() + n,
                   normal_errors.end());
  const double median = normal_errors[n];
  return {mean, median, standard_deviation};
}

/*sauto masterThread() -> void {
  // Read the full graph in such that we know what to expect
  dba::DataUniquePtr data_ptr =
      std::make_unique<dba::Data>(std::numeric_limits<uint64_t>::max());
  CHECK(data_ptr->readDataFromFiles(FLAGS_data_folder));
  dba::Optimization optimizer(data_ptr,
                              dba::Optimization::ConsensusType::kNoConsensus);
  CHECK(optimizer.setupProblem());
  // Allocate a vector large enough to hold all data
  // From the receiver we expect the following structure
  // [num_frames, num_map_points, num_frames * [id, p_x, p_y, p_z, dq_x,
  // dq_y, dq_z, intrinsics, distortion], num_map_points * [id, p_x, p_y,
  // p_z]]
  const size_t num_map_point_vars = 1 + 3;
  size_t num_vars = 2 + data_ptr->getFrameIds().size() * num_frame_vars +
                    data_ptr->getMapPointIds().size() * num_map_point_vars;
  std::vector<double> data_vec(num_vars);
  for (size_t i = 0; i < FLAGS_max_iter_eval; ++i) {
    int iteration = static_cast<int>(i);
    // Send the iteration nr (a.k.a. start command for the processor nodes)
    for (size_t nr = 0; nr < FLAGS_num_subgraphs; ++nr) {
      MPI_Send(&iteration, 1, MPI_INT, nr + 1, kInitiateOffset, MPI_COMM_WORLD);
    }

    // Prepare storage for incoming data (id -> vector of data)
    std::unordered_map<uint64_t, FrameData> frame_vars;
    std::unordered_map<uint64_t, uint64_t> frame_vars_count;
    std::unordered_map<uint64_t, Eigen::Vector3d> map_point_vars;
    std::unordered_map<uint64_t, uint64_t> map_point_vars_count;
    for (size_t nr = 0; nr < FLAGS_num_subgraphs; ++nr) {
      MPI_Status status;
      MPI_Probe(nr + 1, kReceiveOffset, MPI_COMM_WORLD, &status);
      int incoming_vars = 0;
      MPI_Get_count(&status, MPI_DOUBLE, &incoming_vars);
      MPI_Recv(&data_vec, incoming_vars, MPI_DOUBLE, nr + 1, kReceiveOffset,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      const size_t num_frames = static_cast<size_t>(data_vec[0] + kEps);
      const size_t num_map_points = static_cast<size_t>(data_vec[1] + kEps);

      // Update the frame variables (sum)
      for (size_t fi = 0; fi < num_frames; ++fi) {
        const size_t offset = 2 + fi * num_frame_vars;
        const uint64_t id = static_cast<uint64_t>(data_vec[offset] + kEps);
        auto find_itr = frame_vars.find(id);
        if (find_itr == frame_vars.end()) {
          FrameData zero_var(id);
          frame_vars.insert({id, zero_var});
          find_itr = frame_vars.find(id);
        }
        auto& avg = find_itr->second;
        for (size_t vi = 0; vi < num_frame_vars - 1; ++vi) {
          const size_t var_data_offset = offset + 1;  // skip id
          avg[vi] = avg[vi] + data_vec[var_data_offset + vi];
        }
        ++frame_vars_count[id];
      }

      // Update the map point variables (sum)
      for (size_t mi = 0; mi < num_map_points; ++mi) {
        const size_t offset = 2 + num_frames * num_frame_vars + mi * 4;
        const uint64_t id = static_cast<uint64_t>(data_vec[offset] + kEps);
        auto find_itr = map_point_vars.find(id);
        if (find_itr == map_point_vars.end()) {
          Eigen::Vector3d zero_var = Eigen::Vector3d::Zero();
          map_point_vars.insert({id, zero_var});
          find_itr = map_point_vars.find(id);
        }
        auto& avg = find_itr->second;
        for (size_t vi = 0; vi < 3; ++vi) {
          const size_t var_data_offset = offset + 1;
          avg[vi] = avg[vi] + data_vec[var_data_offset];
        }
        ++map_point_vars_count[id];
      }
    }

    // Compute the actual values (averaging) and update the varible in the
    // data_ptr for the error computation
    for (auto& [id, var] : frame_vars) {
      // Compute the average
      const auto count = frame_vars_count[id];
      for (size_t k = 0; k < var.getSize(); ++i) {
        var[k] = var[k] / count;
      }

      // Set the actual variables in the frame
      auto frame_ptr = data_ptr->getFrame(id);
      CHECK(frame_ptr != nullptr);
      Eigen::Vector3d delta_q;
      for (size_t k = 0; k < FrameData::getSize(); ++k) {
        if (k < 3) {
          // Translation
          frame_ptr->p_W_C_[k] = var[k];
        } else if (k >= 3 && k < 6) {
          // Rotation
          delta_q[k - 3] = var[k];
        } else if (k >= 6 && k < 6 + kNumIntrinsicParams) {
          // Intrinsics
          frame_ptr->intrinsics_[k - 6] = var[k];
        } else {
          // Distortion
          frame_ptr->dist_coeffs_[k - 6 - kNumIntrinsicParams] = var[k];
        }
      }
      dba::utils::rotmath::Plus(frame_ptr->getReferenceRotation(), delta_q,
                                &frame_ptr->q_W_C_);
    }
    for (auto& [id, var] : map_point_vars) {
      const auto count = map_point_vars_count[id];
      var = var / count;
      auto map_point_ptr = data_ptr->getMapPoint(id);
      CHECK(map_point_ptr != nullptr);
      map_point_ptr->position_ = var;
    }

    // Now we can compute the errors
    const auto errors = optimizer.computeErrors();
  }
}*/

// Function to evaluate a range of iterations (parallelized by iterations)
auto computeErrorWorker(const int worker_id, const size_t start_itr,
                        const size_t end_itr) -> void {
  std::cout << "Worker: " << worker_id << ", start: " << start_itr
            << ", end: " << end_itr << std::endl;
  // CHECK_GE(end_itr, start_itr);
  std::string pre_app;
  if (FLAGS_admm_type == "central_sync") {
    pre_app = "_opt_cent_sync_";
  } else if (FLAGS_admm_type == "central_async") {
    pre_app = "_opt_cent_async_";
  } else if (FLAGS_admm_type == "decentral_async") {
    pre_app = "_opt_decent_async_";
  } else {
    LOG(FATAL) << "Unknown parameter for admm_type (" << FLAGS_admm_type << ")";
  }

  // Read the full graph in such that we know what to expect
  dba::DataSharedPtr data_ptr =
      std::make_shared<dba::Data>(std::numeric_limits<uint64_t>::max());
  CHECK(data_ptr->readDataFromFiles(FLAGS_data_folder));
  dba::Optimization optimizer(data_ptr,
                              dba::Optimization::ConsensusType::kNoConsensus);
  CHECK(optimizer.setupProblem());

  // Wait for the start signal from the master
  int start_signal;
  MPI_Recv(&start_signal, 1, MPI_INT, 0, worker_id + tagoffset_start,
           MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  std::vector<double> result;
  result.reserve((end_itr - start_itr + 1) * 4);
  for (size_t iter = start_itr; iter <= end_itr; ++iter) {
    std::unordered_map<uint64_t, FrameData> frame_vars;
    std::unordered_map<uint64_t, uint64_t> frame_vars_count;
    std::unordered_map<uint64_t, Eigen::Vector3d> map_point_vars;
    std::unordered_map<uint64_t, uint64_t> map_point_vars_count;
    for (size_t nr = 0; nr < FLAGS_num_subgraphs; ++nr) {
      const std::string app = pre_app + std::to_string(iter);
      dba::Data local_data(nr);
      //      if (iter == 0) {
      //        CHECK(local_data.readDataFromFiles(FLAGS_data_folder));
      //      } else {
      CHECK(local_data.readDataFromFiles(FLAGS_result_folder, app));
      //      }
      const auto& frame_ids = local_data.getFrameIds();
      for (const auto& f_id : frame_ids) {
        auto ref_frame_ptr = data_ptr->getFrame(f_id);
        CHECK(ref_frame_ptr != nullptr);
        auto frame_ptr = local_data.getFrame(f_id);
        CHECK(frame_ptr != nullptr);
        auto find_itr = frame_vars.find(f_id);
        if (find_itr == frame_vars.end()) {
          FrameData zero_var(f_id);
          frame_vars.insert({f_id, zero_var});
          find_itr = frame_vars.find(f_id);
        }
        auto& avg = find_itr->second;
        Eigen::Vector3d delta_q;
        dba::utils::rotmath::Minus(
            frame_ptr->q_W_C_, ref_frame_ptr->getReferenceRotation(), &delta_q);
        for (size_t k = 0; k < FrameData::getSize(); ++k) {
          if (k < 3) {
            // Translation
            avg[k] = avg[k] + frame_ptr->p_W_C_[k];
          } else if (k >= 3 && k < 6) {
            // Rotation
            avg[k] = avg[k] + delta_q[k - 3];
          } else if (k >= 6 && k < 6 + kNumIntrinsicParams) {
            // Intrinsics
            avg[k] = avg[k] + frame_ptr->intrinsics_[k - 6];
          } else {
            // Distortion
            avg[k] =
                avg[k] + frame_ptr->dist_coeffs_[k - 6 - kNumIntrinsicParams];
          }
        }
        ++frame_vars_count[f_id];
      }

      const auto& map_point_ids = local_data.getMapPointIds();
      for (const auto& mp_id : map_point_ids) {
        auto map_point_ptr = local_data.getMapPoint(mp_id);
        CHECK(map_point_ptr != nullptr);
        auto find_itr = map_point_vars.find(mp_id);
        if (find_itr == map_point_vars.end()) {
          Eigen::Vector3d zero_var = Eigen::Vector3d::Zero();
          map_point_vars.insert({mp_id, zero_var});
          find_itr = map_point_vars.find(mp_id);
        }
        auto& avg = find_itr->second;
        Eigen::Vector3d delta_q;
        avg = avg + map_point_ptr->position_;
        ++map_point_vars_count[mp_id];
      }
    }

    // Compute the actual values (averaging) and update the varible in the
    // data_ptr for the error computation
    for (auto& [id, var] : frame_vars) {
      // Compute the average
      const auto count = frame_vars_count[id];
      for (size_t k = 0; k < var.getSize(); ++k) {
        var[k] = var[k] / static_cast<double>(count);
      }

      // Set the actual variables in the frame
      auto frame_ptr = data_ptr->getFrame(id);
      CHECK(frame_ptr != nullptr);
      Eigen::Vector3d delta_q;
      for (size_t k = 0; k < FrameData::getSize(); ++k) {
        if (k < 3) {
          // Translation
          frame_ptr->p_W_C_[k] = var[k];
        } else if (k >= 3 && k < 6) {
          // Rotation
          delta_q[k - 3] = var[k];
        } else if (k >= 6 && k < 6 + kNumIntrinsicParams) {
          // Intrinsics
          frame_ptr->intrinsics_[k - 6] = var[k];
        } else {
          // Distortion
          frame_ptr->dist_coeffs_[k - 6 - kNumIntrinsicParams] = var[k];
        }
      }
      dba::utils::rotmath::Plus(frame_ptr->getReferenceRotation(), delta_q,
                                &frame_ptr->q_W_C_);
    }
    for (const auto& [id, var] : map_point_vars) {
      const auto count = map_point_vars_count[id];
      auto map_point_ptr = data_ptr->getMapPoint(id);
      CHECK(map_point_ptr != nullptr);
      map_point_ptr->position_ = var / static_cast<double>(count);
    }

    // Now we can compute the errors
    const auto errors = optimizer.computeErrors();
    const auto& [mean, median, standard_deviation] = computeMetrics(errors);
    result.push_back(static_cast<double>(iter));
    result.push_back(mean);
    result.push_back(median);
    result.push_back(standard_deviation);
  }

  // Send the data to the master node
  MPI_Send(result.data(), result.size(), MPI_DOUBLE, 0,
           worker_id + tagoffset_state_to_master, MPI_COMM_WORLD);
}

auto computeErrorMaster(const std::vector<std::pair<size_t, size_t>>& intervals)
    -> void {
  // Give start signal to workers
  int start_signal = 1;
  for (int worker_id = 1; worker_id < intervals.size() + 1; ++worker_id) {
    if (intervals[worker_id - 1].first <= intervals[worker_id - 1].second) {
      MPI_Send(&start_signal, 1, MPI_INT, worker_id,
               worker_id + tagoffset_start, MPI_COMM_WORLD);
    }
  }
  // Open file
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

  // Compute the initial error
  dba::DataSharedPtr data_ptr =
      std::make_shared<dba::Data>(std::numeric_limits<uint64_t>::max());
  CHECK(data_ptr->readDataFromFiles(FLAGS_data_folder));
  dba::Optimization optimizer(data_ptr,
                              dba::Optimization::ConsensusType::kNoConsensus);
  CHECK(optimizer.setupProblem());
  const auto errors = optimizer.computeErrors();
  const auto& [mean_0, median_0, standard_deviation_0] = computeMetrics(errors);

  std::ofstream file;
  file.open(filename);
  file << -1 << "," << mean_0 << "," << median_0 << "," << standard_deviation_0
       << std::endl;
  for (size_t i = 0; i < intervals.size(); ++i) {
    if (intervals[i].second < intervals[i].first) continue;
    const int worker_id = static_cast<int>(i) + 1;
    std::vector<double> data;
    data.resize((intervals[i].second - intervals[i].first + 1) * 4);
    MPI_Recv(data.data(), data.size(), MPI_DOUBLE, worker_id,
             worker_id + tagoffset_state_to_master, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
    for (size_t j = 0; j < data.size(); j += 4) {
      const size_t iteration = static_cast<double>(data[j] + kEps);
      const double mean = data[j + 1];
      const double median = data[j + 2];
      const double standard_deviation = data[j + 3];
      std::cout << iteration << ", " << mean << ", " << median << ", "
                << standard_deviation_0 << std::endl;
      file << iteration << "," << mean << "," << median << ","
           << standard_deviation << std::endl;
    }
  }
  file.close();
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  CHECK(FLAGS_admm_type == "central_sync" ||
        FLAGS_admm_type == "central_async" ||
        FLAGS_admm_type == "decentral_async");
  MPI_Init(&argc, &argv);
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Compute the intervals
  const size_t num_intervals = world_size - 1;
  const size_t num_iter = FLAGS_max_iter_eval + 1;
  const double n_interval =
      static_cast<double>(num_iter) / static_cast<double>(num_intervals);
  CHECK_GT(n_interval, 0);
  std::vector<std::pair<size_t, size_t>> intervals;
  intervals.reserve(num_iter);
  double ind = 0.0;
  for (size_t i = 0; i < num_intervals; ++i) {
    if (i + 1 < num_intervals) {
      const double ub = std::max(0.0, ind + n_interval - 1.0);
      intervals.push_back({std::round(ind), std::round(ub)});
      ind = ind + n_interval;
    } else {
      intervals.push_back({static_cast<double>(ind), FLAGS_max_iter_eval});
    }
  }

  if (rank == 0) {
    computeErrorMaster(intervals);
  } else {
    if (intervals[rank - 1].first <= intervals[rank - 1].second) {
      computeErrorWorker(rank, intervals[rank - 1].first,
                         intervals[rank - 1].second);
    }
  }

  MPI_Finalize();
  return 0;
}
