#include <gflags/gflags.h>
#include <mpi.h>
#include <experimental/filesystem>
#include <fstream>
#include <iomanip>

#include "distributed_bundle_adjustment/central_communication.hpp"
#include "distributed_bundle_adjustment/utils.hpp"

DEFINE_uint64(num_admm_iter, 30, "The number of iterations for the ADMM");
DEFINE_uint32(write_out_iter, 3, "Iterations until result gets written out");
DECLARE_string(result_folder);
DECLARE_string(data_folder);

namespace filesystem = std::experimental::filesystem;

namespace dba {

MasterNode::MasterNode(const int num_workers, DataSharedPtr data_ptr)
    : num_workers_(num_workers), data_ptr_(data_ptr), has_started_(false) {}

MasterNode::~MasterNode() { comm_thread_.join(); }

auto MasterNode::startNodes() -> bool {
  if (has_started_) return true;
  int start = 1;
  for (int i = 1; i <= num_workers_; ++i) {
    MPI_Send(&start, 1, MPI_INT, i, 0 + tagoffset_start, MPI_COMM_WORLD);
  }
  comm_thread_ = std::thread(&MasterNode::communicationLoop, this);
}

auto MasterNode::communicationLoop() -> void {
  const std::string timer_filename =
      FLAGS_result_folder + "/iteration_timing_cent_sync.csv";
  std::ofstream timer_file;
  timer_file.open(timer_filename);
  const auto start_time = std::chrono::high_resolution_clock::now();
  for (size_t k = 1; k <= FLAGS_num_admm_iter; ++k) {
    // First extract all the received information and store it locally
    std::unordered_map<uint64_t, std::vector<FrameDual>> incoming_frames;
    std::unordered_map<uint64_t, std::vector<MapPointDual>> incoming_map_points;
    for (size_t i = 1; i <= num_workers_; ++i) {
      // The data that we will receive is [num_frames, num_map_points,
      // num_frames * [frame_id, p_x, p_y, p_x, dq_x, dq_y, dq_z, f, k1, k2],
      // num_map_points * [mp_id, p_x, p_y, p_z]]
      const auto& comm_frames = data_ptr_->getCommFrames(i - 1);
      const auto& comm_map_points = data_ptr_->getCommMapPoints(i - 1);
      const size_t num_vars =
          comm_frames.size() * (FrameDual::getSize() + 1) +
          comm_map_points.size() * (MapPointDual::getSize() + 1) + 2;
      std::vector<double> data_in(num_vars);
      MPI_Recv(data_in.data(), num_vars, MPI_DOUBLE, i,
               i + tagoffset_state_to_master, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
      const size_t num_frames = static_cast<size_t>(data_in[0] + 1e-8);
      CHECK_EQ(num_frames, comm_frames.size()) << "Graph " << i - 1;
      const size_t num_map_points = static_cast<size_t>(data_in[1] + 1e-8);
      CHECK_EQ(num_map_points, comm_map_points.size());
      for (size_t f_nr = 0; f_nr < num_frames; ++f_nr) {
        const size_t start_ind = 2 + f_nr * (FrameDual::getSize() + 1);
        // Check whether this frame has been marked as an outlier. In that case
        // mark it as such in order to avoid updating it.
        const double comm_id = data_in[start_ind] + 1e-8;
        if (comm_id < 0.0) {
          const uint64_t id =
              static_cast<uint64_t>(data_in[start_ind + 1] + 1e-8);
          outlier_vars_.insert(id);
          continue;
        }
        const uint64_t frame_id = static_cast<uint64_t>(comm_id);
        FrameDual dual(frame_id);
        for (size_t j = 0; j < dual.getSize(); ++j)
          dual[j] = data_in[start_ind + j + 1];
        incoming_frames[frame_id].push_back(dual);
      }
      for (size_t mp_nr = 0; mp_nr < num_map_points; ++mp_nr) {
        const size_t start_ind = 2 + num_frames * (FrameDual::getSize() + 1) +
                                 mp_nr * (MapPointDual::getSize() + 1);
        // Check whether this frame has been marked as an outlier. In that case
        // mark it as such in order to avoid updating it.
        const double comm_id = data_in[start_ind] + 1e-8;
        if (comm_id < 0.0) {
          const uint64_t id =
              static_cast<uint64_t>(data_in[start_ind + 1] + 1e-8);
          outlier_vars_.insert(id);
          continue;
        }
        const uint64_t map_point_id = static_cast<uint64_t>(comm_id);
        MapPointDual dual(map_point_id);
        for (size_t j = 0; j < dual.getSize(); ++j)
          dual[j] = data_in[start_ind + j + 1];
        incoming_map_points[map_point_id].push_back(dual);
      }
    }
    LOG(INFO) << "Iteration " << k << " of " << FLAGS_num_admm_iter
              << std::endl;
    LOG(INFO) << "Currently there are " << outlier_vars_.size()
              << " variables classified as outliers";

    // No we have all the data, compute the average over all variables
    std::unordered_map<uint64_t, FrameDual> average_frame;
    for (const auto& [id, vars] : incoming_frames) {
      FrameDual avg(id);
      avg.fill(0);
      const double M = static_cast<double>(vars.size());
      for (const auto& d : vars) {
        for (size_t i = 0; i < d.getSize(); ++i) avg[i] += d[i] / M;
      }
      average_frame[id] = avg;
      auto frame_ptr = data_ptr_->getFrame(id);
      CHECK(frame_ptr != nullptr);
    }
    std::unordered_map<uint64_t, MapPointDual> average_map_point;
    for (const auto& [id, vars] : incoming_map_points) {
      MapPointDual avg(id);
      avg.fill(0);
      const double M = static_cast<double>(vars.size());
      for (const auto& d : vars) {
        for (size_t i = 0; i < d.getSize(); ++i) avg[i] += d[i] / M;
      }
      average_map_point[id] = avg;
    }

    // Communicate the average back to the workers
    // The data that gets send has the format
    // [num_frames, num_map_points, num_frames * [id, p_x, p_y, p_z, dq_x, dq_y,
    // dq_z, f, k1, k2], num_map_points * [id, p_x, p_y, p_z]]
    for (int i = 1; i <= num_workers_; ++i) {
      const auto& frame_data = data_ptr_->getCommFrames(i - 1);
      const auto& map_point_data = data_ptr_->getCommMapPoints(i - 1);
      const size_t num_frames = frame_data.size();
      const size_t num_map_points = map_point_data.size();
      const size_t num_vars = 2 + num_frames * (FrameDual::getSize() + 1) +
                              num_map_points * (MapPointDual::getSize() + 1);
      std::vector<double> data_out(num_vars);
      data_out[0] = static_cast<double>(num_frames);
      data_out[1] = static_cast<double>(num_map_points);
      for (size_t j = 0; j < num_frames; ++j) {
        const size_t start_ind = 2 + j * (FrameDual::getSize() + 1);
        const uint64_t frame_id = frame_data[j];
        if (outlier_vars_.count(frame_id)) {
          data_out[start_ind] = -1.0;
          data_out[start_ind + 1] = static_cast<double>(frame_id);
          continue;
        }
        CHECK(average_frame.count(frame_id))
            << "Cannot find average for frame " << frame_id;
        data_out[start_ind] = static_cast<double>(frame_id);
        const auto& av = average_frame[frame_id];
        for (size_t vi = 0; vi < FrameDual::getSize(); ++vi) {
          data_out[start_ind + 1 + vi] = av[vi];
        }
      }
      for (size_t j = 0; j < num_map_points; ++j) {
        const size_t start_ind = num_frames * (FrameDual::getSize() + 1) +
                                 j * (MapPointDual::getSize() + 1) + 2;
        const uint64_t map_point_id = map_point_data[j];
        if (outlier_vars_.count(map_point_id)) {
          data_out[start_ind] = -1.0;
          data_out[start_ind + 1] = static_cast<double>(map_point_id);
          continue;
        }
        CHECK(average_map_point.count(map_point_id));
        data_out[start_ind] = static_cast<double>(map_point_id);
        const auto& av = average_map_point[map_point_id];
        for (size_t vi = 0; vi < MapPointDual::getSize(); ++vi) {
          data_out[start_ind + 1 + vi] = av[vi];
        }
      }

      MPI_Send(data_out.data(), data_out.size(), MPI_DOUBLE, i,
               i + tagoffset_average_to_worker, MPI_COMM_WORLD);
    }
    const auto time1 = std::chrono::high_resolution_clock::now();
    timer_file << k << ","
               << std::chrono::duration_cast<std::chrono::milliseconds>(
                      time1 - start_time)
                      .count()
               << std::endl;
  }
  timer_file.close();
  this->computeFinalResult();
  //  data_ptr_->writeOutResult("sync_final");
}

auto MasterNode::computeFinalResult() -> void {
  LOG(WARNING) << "At the moment computeFinalResult is not implemented";
  //  std::unordered_map<uint64_t, std::vector<std::array<double, 10>>>
  //      incoming_data;
  //  for (int i = 0; i < num_workers_; ++i) {
  //    // We receive the data as num_frame * [id, p_x, p_y, p_z, q_x, q_y, q_z,
  //    // q_w, f, k1, k2]
  //    const int num_frames = data_ptr_->getFrameIds(i).size();
  //    CHECK_GT(num_frames, 0);
  //    std::vector<double> frames_in(num_frames * 11, 0);
  //    MPI_Recv(frames_in.data(), num_frames * 11, MPI_DOUBLE, i + 1, 0,
  //             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  //    for (int fr = 0; fr < num_frames; ++fr) {
  //      const uint64_t id = static_cast<double>(frames_in[fr * 11]);
  //      std::array<double, 10> tmp;
  //      for (int j = 0; j < 10; ++j) tmp[j] = frames_in[fr * 11 + j + 1];
  //      incoming_data[id].push_back(tmp);
  //    }
  //  }

  //  // Write out the result (and average it if necessary)
  //  if (!filesystem::is_directory(FLAGS_result_folder + "/Full")) {
  //    LOG(WARNING) << "Creates folder at " << FLAGS_result_folder << "/Full";
  //    CHECK(filesystem::create_directory(FLAGS_result_folder + "/Full"));
  //  }
  //  const std::string filename_frames =
  //      FLAGS_result_folder + "/Full/frames_opt_sync.csv";
  //  std::ofstream file;
  //  file.open(filename_frames);
  //  for (const auto& [id, data] : incoming_data) {
  //    if (outlier_frames_.count(id)) continue;
  //    Eigen::Quaterniond q_W_C_ref(data[0][6], data[0][3], data[0][4],
  //                                 data[0][5]);
  //    q_W_C_ref.normalize();
  //    Eigen::Vector3d delta_q_avg = Eigen::Vector3d::Zero();
  //    Eigen::Vector3d p_W_C_avg = Eigen::Vector3d::Zero();
  //    double f_avg = 0;
  //    double k1_avg = 0, k2_avg = 0;
  //    for (const auto& d : data) {
  //      p_W_C_avg[0] += d[0];
  //      p_W_C_avg[1] += d[1];
  //      p_W_C_avg[2] += d[2];
  //      Eigen::Quaterniond q(d[6], d[3], d[4], d[5]);
  //      q.normalize();
  //      Eigen::Vector3d delta;
  //      utils::rotmath::Minus(q, q_W_C_ref, &delta);
  //      delta_q_avg += delta;
  //      f_avg += d[7];
  //      k1_avg += d[8];
  //      k2_avg += d[9];
  //    }
  //    const int N = data.size();
  //    p_W_C_avg /= N;
  //    delta_q_avg /= N;
  //    Eigen::Quaterniond q_W_C_avg;
  //    utils::rotmath::Plus(q_W_C_ref, delta_q_avg, &q_W_C_avg);
  //    f_avg /= N;
  //    k1_avg /= N;
  //    k2_avg /= N;
  //    file << std::setprecision(20) << id << "," << p_W_C_avg[0] << ","
  //         << p_W_C_avg[1] << "," << p_W_C_avg[2] << "," << q_W_C_avg.x() <<
  //         ","
  //         << q_W_C_avg.y() << "," << q_W_C_avg.z() << "," << q_W_C_avg.w() <<
  //         ","
  //         << f_avg << ",x,x,x," << k1_avg << "," << k2_avg << ",x,x,x"
  //         << std::endl;
  //  }
  //  file.close();

  //  std::unordered_map<uint64_t, VectorOfVector3> incoming_mp_data;
  //  for (int i = 0; i < num_workers_; ++i) {
  //    int num_mps = 0;
  //    MPI_Recv(&num_mps, 1, MPI_DOUBLE, i + 1, 0, MPI_COMM_WORLD,
  //             MPI_STATUS_IGNORE);
  //    CHECK_GT(num_mps, 0);
  //    std::vector<double> map_points_in(num_mps * 4, 0);
  //    MPI_Recv(map_points_in.data(), num_mps * 4, MPI_DOUBLE, i + 1, 0,
  //             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  //    for (int mp = 0; mp < num_mps; ++mp) {
  //      const uint64_t id = static_cast<double>(map_points_in[mp * 4]);
  //      Eigen::Vector3d map_point(map_points_in[mp * 4 + 1],
  //                                map_points_in[mp * 4 + 2],
  //                                map_points_in[mp * 4 + 3]);
  //      incoming_mp_data[id].push_back(map_point);
  //    }
  //  }

  //  const std::string filename_mps =
  //      FLAGS_result_folder + "/Full/map_points_opt_sync.csv";
  //  file.open(filename_mps);
  //  for (const auto& [id, data] : incoming_mp_data) {
  //    Eigen::Vector3d position = Eigen::Vector3d::Zero();
  //    for (const auto& d : data) {
  //      position[0] += d[0];
  //      position[1] += d[1];
  //      position[2] += d[2];
  //    }
  //    const int N = data.size();
  //    position /= N;
  //    file << std::setprecision(20) << id << "," << position[0] << ","
  //         << position[1] << "," << position[2] << std::endl;
  //  }
  //  file.close();
}

WorkerNode::WorkerNode(const int rank, DataSharedPtr data_ptr)
    : rank_(rank), data_ptr_(data_ptr) {
  CHECK(data_ptr_ != nullptr);
  optimization_ptr_ = std::make_unique<Optimization>(
      data_ptr_, Optimization::ConsensusType::kCentral);
  CHECK(optimization_ptr_->setupProblem());
  this->startNode();
}

WorkerNode::~WorkerNode() { process_thread_.join(); }

auto WorkerNode::startNode() -> void {
  int command = -1;
  MPI_Recv(&command, 1, MPI_INT, 0, 0 + tagoffset_start, MPI_COMM_WORLD,
           MPI_STATUS_IGNORE);
  process_thread_ = std::thread(&WorkerNode::processLoop, this);
}

auto WorkerNode::processLoop() -> void {
  // Write out initial state
  data_ptr_->writeOutResult("cent_sync_" + std::to_string(0));

  // Create the timing file
  const std::string timing_filename = FLAGS_result_folder + "/Graph_" +
                                      std::to_string(data_ptr_->getGraphId()) +
                                      "/timing_sync.csv";
  std::ofstream timing_file;
  timing_file.open(timing_filename);
  auto time_start = std::chrono::high_resolution_clock::now();
  for (int i = 1; i <= FLAGS_num_admm_iter; ++i) {
    auto time1 = std::chrono::high_resolution_clock::now();
    // Optimize
    optimization_ptr_->performOptimization();
    auto time2 = std::chrono::high_resolution_clock::now();
    timing_file << std::chrono::duration_cast<std::chrono::milliseconds>(
                       time1 - time_start)
                       .count()
                << ","
                << std::chrono::duration_cast<std::chrono::milliseconds>(
                       time2 - time_start)
                       .count()
                << std::endl;
    // Communicate the data to the master for averaging
    this->communicateData();
    if (i % FLAGS_write_out_iter == 0)
      data_ptr_->writeOutResult("cent_sync_" + std::to_string(i));
  }
  const auto& residuals = optimization_ptr_->computeErrors();
  std::string filename = FLAGS_data_folder + "/Graph_" +
                         std::to_string(rank_ - 1) + "/residuals.csv";
  std::ofstream file;
  file.open(filename);
  for (const auto r : residuals) {
    file << r << std::endl;
  }
  file.close();

  //  // Send the frame data to the master
  //  const auto& frame_ids = data_ptr_->getFrameIds();
  //  const int num_frames = frame_ids.size();
  //  std::vector<double> frame_comm_data(num_frames * 11, 0);
  //  for (int fr = 0; fr < num_frames; ++fr) {
  //    auto frame_ptr = data_ptr_->getFrame(frame_ids[fr]);
  //    frame_comm_data[fr * 11] = frame_ids[fr];
  //    frame_comm_data[fr * 11 + 1] = frame_ptr->p_W_C_[0];
  //    frame_comm_data[fr * 11 + 2] = frame_ptr->p_W_C_[1];
  //    frame_comm_data[fr * 11 + 3] = frame_ptr->p_W_C_[2];

  //    frame_comm_data[fr * 11 + 4] = frame_ptr->q_W_C_.x();
  //    frame_comm_data[fr * 11 + 5] = frame_ptr->q_W_C_.y();
  //    frame_comm_data[fr * 11 + 6] = frame_ptr->q_W_C_.z();
  //    frame_comm_data[fr * 11 + 7] = frame_ptr->q_W_C_.w();

  //    frame_comm_data[fr * 11 + 8] = frame_ptr->intrinsics_[0];

  //    frame_comm_data[fr * 11 + 9] = frame_ptr->dist_coeffs_[0];
  //    frame_comm_data[fr * 11 + 10] = frame_ptr->dist_coeffs_[1];
  //  }
  //  MPI_Send(frame_comm_data.data(), frame_comm_data.size(), MPI_DOUBLE, 0, 0,
  //           MPI_COMM_WORLD);

  //  // Send the map points to the master
  //  const auto& mp_ids = data_ptr_->getMapPointIds();
  //  const int num_map_points = mp_ids.size();
  //  MPI_Send(&num_map_points, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
  //  std::vector<double> mp_comm_data(num_map_points * 4, 0);
  //  for (int mp = 0; mp < num_map_points; ++mp) {
  //    auto mp_ptr = data_ptr_->getMapPoint(mp_ids[mp]);
  //    CHECK(mp_ptr != nullptr);
  //    mp_comm_data[mp * 4] = mp_ids[mp];
  //    mp_comm_data[mp * 4 + 1] = mp_ptr->position_[0];
  //    mp_comm_data[mp * 4 + 2] = mp_ptr->position_[1];
  //    mp_comm_data[mp * 4 + 3] = mp_ptr->position_[2];
  //  }
  //  MPI_Send(mp_comm_data.data(), mp_comm_data.size(), MPI_DOUBLE, 0, 0,
  //           MPI_COMM_WORLD);
}

auto WorkerNode::communicateData() -> void {
  // Prepare the data for communication to the master node
  const auto& frame_ids = data_ptr_->getCommFrames(rank_ - 1);
  const auto& map_point_ids = data_ptr_->getCommMapPoints(rank_ - 1);
  const size_t num_frames = frame_ids.size();
  const size_t num_map_points = map_point_ids.size();

  // Extract and send the local variables The data has the format
  // [num_frames, num_map_point, num_frames * [id, p_x, p_y, p_z, dq_x, dq_y,
  // dq_z, intrinsics, distortion], num_map_points * [id, p_x, p_y, p_z]] Fill
  // in the frame data
  const size_t num_vars = 2 + num_frames * (FrameDual::getSize() + 1) +
                          num_map_points * (MapPointDual::getSize() + 1);
  std::vector<double> communication_data(num_vars);
  communication_data[0] = static_cast<double>(num_frames);
  communication_data[1] = static_cast<double>(num_map_points);
  for (size_t i = 0; i < num_frames; ++i) {
    const size_t start_ind = 2 + i * (FrameDual::getSize() + 1);
    const uint64_t frame_id = frame_ids[i];
    auto frame_ptr = data_ptr_->getFrame(frame_id);
    CHECK(frame_ptr != nullptr);
    // If this frame is classified as an outlier by the worker, communicate this
    // to the master. This is done by sending a negative value at the id
    // position and the frame id at the subsequent slot.
    if (!frame_ptr->is_valid_) {
      communication_data[start_ind] = -1.0;
      communication_data[start_ind + 1] = static_cast<double>(frame_id);
      continue;
    }
    communication_data[start_ind] = static_cast<double>(frame_id);
    Eigen::Vector3d delta_q;
    utils::rotmath::Minus(frame_ptr->q_W_C_, frame_ptr->getReferenceRotation(),
                          &delta_q);
    for (size_t j = 0; j < FrameDual::getSize(); ++j) {
      if (j < 3) {
        // Translation
        communication_data[start_ind + 1 + j] = frame_ptr->p_W_C_[j];
      } else if (j >= 3 && j < 6) {
        // Rotation
        communication_data[start_ind + 1 + j] = delta_q[j - 3];
      } else if (j >= 6 && j < 6 + kNumIntrinsicParams) {
        // Intrinsics
        communication_data[start_ind + 1 + j] = frame_ptr->intrinsics_[j - 6];
      } else {
        // Distortion
        communication_data[start_ind + 1 + j] =
            frame_ptr->dist_coeffs_[j - 6 - kNumIntrinsicParams];
      }
    }
  }

  // Fill in the map point data
  for (size_t i = 0; i < num_map_points; ++i) {
    const size_t start_ind = 2 + num_frames * (FrameDual::getSize() + 1) +
                             i * (MapPointDual::getSize() + 1);
    const uint64_t map_point_id = map_point_ids[i];
    auto map_point_ptr = data_ptr_->getMapPoint(map_point_id);
    CHECK(map_point_ptr != nullptr);
    // If this map point is classified as an outlier by the worker, communicate
    // this to the master. This is done by sending a negative value at the id
    // position and the frame id at the subsequent slot.
    if (!map_point_ptr->is_valid_) {
      communication_data[start_ind] = -1.0;
      communication_data[start_ind + 1] = static_cast<double>(map_point_id);
      continue;
    }
    communication_data[start_ind] = static_cast<double>(map_point_id);
    for (size_t j = 0; j < MapPointDual::getSize(); ++j) {
      communication_data[start_ind + 1 + j] = map_point_ptr->position_[j];
    }
  }
  MPI_Send(communication_data.data(), communication_data.size(), MPI_DOUBLE, 0,
           rank_ + tagoffset_state_to_master, MPI_COMM_WORLD);

  // Receive and extract the averaged result
  // [num_frames, num_map_point, num_frames * [id, p_x, p_y, p_z, dq_x, dq_y,
  // dq_z, intrinsics, distortion], num_map_points * [id, p_x, p_y, p_z]]
  std::fill(communication_data.begin(), communication_data.begin(), 0.0);
  MPI_Recv(communication_data.data(), communication_data.size(), MPI_DOUBLE, 0,
           rank_ + tagoffset_average_to_worker, MPI_COMM_WORLD,
           MPI_STATUS_IGNORE);
  const size_t recv_num_frames = static_cast<double>(communication_data[0]);
  CHECK_EQ(recv_num_frames, num_frames);
  const size_t recv_num_map_points = static_cast<double>(communication_data[1]);
  CHECK_EQ(recv_num_map_points, num_map_points);

  // First extract the frame data
  std::unordered_map<uint64_t, FrameDual> incoming_frames;
  for (size_t i = 0; i < num_frames; ++i) {
    // Check whether this frame is classified as outlier. This can be the case
    // if it was classified by this node or any of the neighboring nodes in
    // the last iteration.
    const size_t start_ind = 2 + i * (FrameDual::getSize() + 1);
    const double recv_id = communication_data[start_ind] + 1e-8;
    if (recv_id < 0.0) {
      const uint64_t frame_id =
          static_cast<uint64_t>(communication_data[start_ind + 1]);
      auto frame_ptr = data_ptr_->getFrame(frame_id);
      CHECK(frame_ptr != nullptr);
      frame_ptr->is_valid_ = false;
      continue;
    }

    const uint64_t frame_id = static_cast<uint64_t>(recv_id);
    FrameDual tmp_var(frame_id);
    for (size_t j = 0; j < tmp_var.getSize(); ++j)
      tmp_var[j] = communication_data[start_ind + 1 + j];
    incoming_frames[frame_id] = tmp_var;
  }

  std::unordered_map<uint64_t, MapPointDual> incoming_map_points;
  for (size_t i = 0; i < num_map_points; ++i) {
    // Check whether this frame is classified as outlier. This can be the case
    // if it was classified by this node or any of the neighboring nodes in
    // the last iteration.
    const size_t start_ind = 2 + num_frames * (FrameDual::getSize() + 1) +
                             i * (MapPointDual::getSize() + 1);
    const double recv_id = communication_data[start_ind] + 1e-8;
    if (recv_id < 0.0) {
      const uint64_t map_point_id =
          static_cast<uint64_t>(communication_data[start_ind + 1]);
      auto map_point_ptr = data_ptr_->getMapPoint(map_point_id);
      CHECK(map_point_ptr != nullptr);
      map_point_ptr->is_valid_ = false;
      continue;
    }

    const uint64_t map_point_id = static_cast<uint64_t>(recv_id);
    MapPointDual tmp_var(map_point_id);
    for (size_t j = 0; j < tmp_var.getSize(); ++j)
      tmp_var[j] = communication_data[start_ind + 1 + j];
    incoming_map_points[map_point_id] = tmp_var;
  }
  CHECK(optimization_ptr_->updateAverages(incoming_frames, incoming_map_points,
                                          true));
}

}  // namespace dba
