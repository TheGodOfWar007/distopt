#include <gflags/gflags.h>
#include <mpi.h>
#include <chrono>
#include <experimental/filesystem>
#include <fstream>
#include <iomanip>

#include "distributed_bundle_adjustment/central_async_communication.hpp"
#include "distributed_bundle_adjustment/utils.hpp"

DEFINE_uint32(num_async, 1,
              "The number of received variables necessary to trigger an "
              "consensus update");
DECLARE_uint64(num_admm_iter);
DECLARE_uint32(write_out_iter);
DECLARE_string(result_folder);
DECLARE_string(data_folder);

namespace filesystem = std::experimental::filesystem;

namespace dba {

AsyncMasterNode::AsyncMasterNode(const int num_workers, DataSharedPtr data_ptr)
    : num_workers_(num_workers),
      data_ptr_(data_ptr),
      has_started_(false),
      has_data_(false),
      iteration_counter_(0) {}

AsyncMasterNode::~AsyncMasterNode() {
  comm_thread_.join();
  for (size_t i = 0; i < num_workers_; ++i) receiver_threads_[i].join();
}

auto AsyncMasterNode::startNodes() -> bool {
  if (has_started_) return true;
  std::unique_lock<std::mutex> lock(start_mutex_);
  int start = 1;
  receiver_threads_.resize(num_workers_);
  for (int i = 1; i <= num_workers_; ++i) {
    receiver_threads_[i - 1] =
        std::thread(&AsyncMasterNode::receiverThread, this, i - 1);
    MPI_Send(&start, 1, MPI_INT, i, tagoffset_start + i, MPI_COMM_WORLD);
  }
  comm_thread_ = std::thread(&AsyncMasterNode::communicationLoop, this);
  has_started_ = true;
  lock.unlock();
  start_cv_.notify_all();
}

auto AsyncMasterNode::communicationLoop() -> void {
  // Create the timing file
  const std::string timer_filename =
      FLAGS_result_folder + "/iteration_timing_cent_async.csv";
  std::ofstream timer_file;
  timer_file.open(timer_filename);
  const auto start_time = std::chrono::high_resolution_clock::now();
  const int num_iterations =
      FLAGS_num_admm_iter * num_workers_ / FLAGS_num_async;
  for (iteration_counter_ = 0; iteration_counter_ < num_iterations;
       ++iteration_counter_) {
    // Wait until enough variables from the neighbors are gathered
    std::unique_lock<std::mutex> lock(data_mutex_);
    incoming_cv_.wait(lock, [this] { return has_data_; });
    const size_t num_incoming =
        std::max(incoming_frames_.size(), incoming_map_points_.size());
    CHECK_EQ(num_incoming, FLAGS_num_async);

    // Update the averages and the corresponding duals
    updateVariables();

    // Communicate the data back to the involved worker nodes
    communicateUpdates();

    const auto time1 = std::chrono::high_resolution_clock::now();
    timer_file << iteration_counter_ + 1 << ","
               << std::chrono::duration_cast<std::chrono::milliseconds>(
                      time1 - start_time)
                      .count()
               << std::endl;

    // Communicate the iteration count to the nodes
    for (size_t n = 0; n < num_workers_; ++n) {
      int count = static_cast<int>(iteration_counter_) + 1;
      MPI_Send(&count, 1, MPI_INT, n + 1, (n + 1) + tagoffset_counter,
               MPI_COMM_WORLD);
    }

    incoming_frames_.clear();
    incoming_map_points_.clear();
    has_data_ = false;
    lock.unlock();
    incoming_cv_.notify_all();
  }
  //  this->computeFinalResult();
  //  data_ptr_->writeOutResult("sync_final");
}

auto AsyncMasterNode::receiverThread(const uint64_t neigh_id) -> void {
  {
    std::unique_lock<std::mutex> lock(start_mutex_);
    start_cv_.wait(lock, [this] { return has_started_; });
    lock.unlock();
    start_cv_.notify_all();
  }
  const int num_iterations =
      FLAGS_num_admm_iter * num_workers_ / FLAGS_num_async;
  while (iteration_counter_ < num_iterations) {
    const auto& frame_ids = data_ptr_->getCommFrames(neigh_id);
    const auto& map_point_ids = data_ptr_->getCommMapPoints(neigh_id);
    CHECK(!frame_ids.empty() || !map_point_ids.empty());
    const size_t num_frames = frame_ids.size();
    const size_t num_map_points = map_point_ids.size();

    // We expect to receive data of the following format
    // [num_frames, num_map_point, num_frames * [f_id, p_x, p_y, p_z, dq_x,
    // dq_y, dq_z, intrinsics, distortion], num_map_points * [mp_id, p_x, p_y,
    // p_z]]
    const size_t num_vars = 2 + num_frames * (FrameDual::getSize() + 1) +
                            num_map_points * (MapPointDual::getSize() + 1);
    std::vector<double> data_in(num_vars);
    MPI_Request request;
    auto val = MPI_Irecv(data_in.data(), num_vars, MPI_DOUBLE, neigh_id + 1,
                         (neigh_id + 1) + tagoffset_state_to_master,
                         MPI_COMM_WORLD, &request);
    int flag = 0;
    MPI_Status status;
    MPI_Test(&request, &flag, &status);
    while (iteration_counter_ < num_iterations && flag == 0) {
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
      MPI_Test(&request, &flag, &status);
    }
    if (flag == 0) {
      MPI_Cancel(&request);
      MPI_Request_free(&request);
      break;
    }
    CHECK(val == MPI_SUCCESS);
    CHECK_EQ(static_cast<size_t>(data_in[0]), num_frames)
        << "from: " << neigh_id << ", data_in " << data_in[0] << ", "
        << data_in[1];
    CHECK_EQ(static_cast<size_t>(data_in[1]), num_map_points);
    std::vector<FrameDual> frame_variables(num_frames);
    for (size_t i = 0; i < num_frames; ++i) {
      const size_t start_ind = 2 + i * (FrameDual::getSize() + 1);
      if (data_in[start_ind] + 1e-10 < 0) {
        // Data was classified as an outlier.
        LOG(FATAL) << "This should not happen at the moment";
        outlier_variables_.insert(
            static_cast<uint64_t>(data_in[start_ind + 1]));
        continue;
      }
      const uint64_t frame_id = static_cast<double>(data_in[start_ind] + 1e-10);
      frame_variables[i].setId(frame_id);
      for (size_t j = 0; j < FrameDual::getSize(); ++j)
        frame_variables[i][j] = data_in[start_ind + 1 + j];
    }
    std::vector<MapPointDual> map_point_variables(num_map_points);
    for (size_t i = 0; i < num_map_points; ++i) {
      const size_t start_ind = 2 + num_frames * (FrameDual::getSize() + 1) +
                               i * (MapPointDual::getSize() + 1);
      if (data_in[start_ind] + 1e-10 < 0) {
        // Data was classified as an outlier.
        LOG(FATAL) << "This should not happen at the moment";
        outlier_variables_.insert(
            static_cast<uint64_t>(data_in[start_ind + 1]));
        continue;
      }
      const uint64_t map_point_id = static_cast<double>(data_in[start_ind]);
      map_point_variables[i].setId(map_point_id);
      for (size_t j = 0; j < MapPointDual::getSize(); ++j)
        map_point_variables[i][j] = data_in[start_ind + 1 + j];
    }
    {
      std::unique_lock<std::mutex> lock(data_mutex_);
      incoming_cv_.wait(lock, [this] { return !has_data_; });
      incoming_frames_[neigh_id] = frame_variables;
      incoming_map_points_[neigh_id] = map_point_variables;
      if (incoming_frames_.size() == FLAGS_num_async) has_data_ = true;
      lock.unlock();
      incoming_cv_.notify_all();
    }

    // Wait until we receive the go for the averaged data and send it back to
    // the workers
    {
      std::unique_lock<std::mutex> lock(data_mutex_);
      incoming_cv_.wait(lock, [this] { return !has_data_; });
      lock.unlock();
      incoming_cv_.notify_all();
    }
  }
}

auto AsyncMasterNode::updateVariables() -> void {
  std::unordered_map<uint64_t, std::vector<std::pair<uint64_t, FrameDual>>>
      frame_variables;
  std::unordered_map<uint64_t, std::vector<std::pair<uint64_t, MapPointDual>>>
      map_point_variables;
  for (const auto& [neigh_id, variable_vec] : incoming_frames_) {
    for (const auto& var : variable_vec) {
      const uint64_t id = var.getId();
      if (outlier_variables_.count(id)) {
        LOG(FATAL) << "This should not happen at the moment";
        continue;
      }
      frame_variables[id].push_back({neigh_id, var});
    }
  }
  for (const auto& [neigh_id, variable_vec] : incoming_map_points_) {
    for (const auto& var : variable_vec) {
      const uint64_t id = var.getId();
      if (outlier_variables_.count(id)) {
        LOG(FATAL) << "This should not happen at the moment";
        continue;
      }
      map_point_variables[id].push_back({neigh_id, var});
    }
  }

  // Update the frame duals and averages
  for (const auto& [id, vars] : frame_variables) {
    const auto num_neighs = vars.size();
    auto frame_ptr = data_ptr_->getFrame(id);
    CHECK(frame_ptr != nullptr);
    frame_ptr->average_state_.fill(0);
    const double sigma_translation = frame_ptr->sigma_trans_;
    const double sigma_rotation = frame_ptr->sigma_rot_;
    const double sigma_intrinsics = frame_ptr->sigma_intr_;
    const double sigma_distortion = frame_ptr->sigma_dist_;
    for (const auto& [n_id, d] : vars) {
      FrameDual dual_variable(id);
      CHECK(frame_ptr->getDualData(n_id, dual_variable));
      for (size_t i = 0; i < dual_variable.getSize(); ++i) {
        if (i < 3) {
          // Translation
          frame_ptr->average_state_[i] +=
              (d[i] + dual_variable[i] / sigma_translation) / num_neighs;
        } else if (i >= 3 && i < 6) {
          // Rotation
          frame_ptr->average_state_[i] +=
              (d[i] + dual_variable[i] / sigma_rotation) / num_neighs;
        } else if (i >= 6 && i < 6 + kNumIntrinsicParams) {
          // Intrinsics
          frame_ptr->average_state_[i] +=
              (d[i] + dual_variable[i] / sigma_intrinsics) / num_neighs;
        } else {
          // Distortion
          frame_ptr->average_state_[i] +=
              (d[i] + dual_variable[i] / sigma_distortion) / num_neighs;
        }
      }
    }

    // Update the dual variable
    for (const auto& [n_id, d] : vars) {
      FrameDual dual_variable;
      CHECK(frame_ptr->getDualData(n_id, dual_variable));
      for (size_t i = 0; i < dual_variable.getSize(); ++i) {
        if (i < 3) {
          // Translation
          dual_variable[i] +=
              sigma_translation * (d[i] - frame_ptr->average_state_[i]);
        } else if (i >= 3 && i < 6) {
          // Rotation

          dual_variable[i] +=
              sigma_rotation * (d[i] - frame_ptr->average_state_[i]);
        } else if (i >= 6 && i < 6 + kNumIntrinsicParams) {
          // Intrinsics
          dual_variable[i] +=
              sigma_intrinsics * (d[i] - frame_ptr->average_state_[i]);
        } else {
          // Distortion
          dual_variable[i] +=
              sigma_distortion * (d[i] - frame_ptr->average_state_[i]);
        }
      }
      frame_ptr->setDualData(n_id, dual_variable);
    }
  }

  // Update the map point duals and averages
  for (const auto& [id, vars] : map_point_variables) {
    const auto num_neighs = vars.size();
    auto map_point_ptr = data_ptr_->getMapPoint(id);
    CHECK(map_point_ptr != nullptr);
    map_point_ptr->average_state_.fill(0);
    const double sigma_map_points = map_point_ptr->sigma_;
    for (const auto& [n_id, d] : vars) {
      MapPointDual dual_variable(id);
      CHECK(map_point_ptr->getDualData(n_id, dual_variable));
      for (size_t i = 0; i < dual_variable.getSize(); ++i) {
        map_point_ptr->average_state_[i] +=
            (d[i] + dual_variable[i] / sigma_map_points) / num_neighs;
      }
    }

    // Update the dual variable
    for (const auto& [n_id, d] : vars) {
      MapPointDual dual_variable;
      CHECK(map_point_ptr->getDualData(n_id, dual_variable));
      for (size_t i = 0; i < dual_variable.getSize(); ++i) {
        dual_variable[i] +=
            sigma_map_points * (d[i] - map_point_ptr->average_state_[i]);
      }
      map_point_ptr->setDualData(n_id, dual_variable);
    }
  }
}

auto AsyncMasterNode::communicateUpdates() -> void {
  std::vector<uint64_t> incoming_ids;
  incoming_ids.reserve(incoming_frames_.size());
  for (const auto& v : incoming_frames_) {
    incoming_ids.push_back(v.first);
  }
  for (const auto neigh_id : incoming_ids) {
    const auto& frame_ids = data_ptr_->getCommFrames(neigh_id);
    const size_t num_frames = frame_ids.size();
    const auto& map_point_ids = data_ptr_->getCommMapPoints(neigh_id);
    const size_t num_map_points = map_point_ids.size();

    // Prepare the communication data with the following ordering in the vector
    // [num_frames, num_map_points, num_frames * [frame_id, avg(trans, rot,
    // intr, dist), dual(trans, rot, intr, dist)], num_map_points *
    // [map_point_id, avg(pos), dual(pos)]]
    const size_t num_vars = 2 + num_frames * (1 + 2 * FrameDual::getSize()) +
                            num_map_points * (1 + 2 * MapPointDual::getSize());
    std::vector<double> data_out(num_vars);
    data_out[0] = static_cast<double>(num_frames);
    data_out[1] = static_cast<double>(num_map_points);

    // Fill in the frame data
    for (size_t i = 0; i < num_frames; ++i) {
      const auto id = frame_ids[i];
      auto frame_ptr = data_ptr_->getFrame(id);
      CHECK(frame_ptr != nullptr);
      const size_t start_ind = 2 + i * (1 + 2 * FrameDual::getSize());
      data_out[start_ind] = static_cast<double>(id);
      // Averaged data
      for (size_t j = 0; j < frame_ptr->average_state_.getSize(); ++j) {
        data_out[start_ind + 1 + j] = frame_ptr->average_state_[j];
      }
      // Dual variables
      FrameDual dual(id);
      CHECK(frame_ptr->getDualData(neigh_id, dual));
      for (size_t j = 0; j < dual.getSize(); ++j) {
        data_out[start_ind + 1 + dual.getSize() + j] = dual[j];
      }
    }

    // Fill in the map point data
    for (size_t i = 0; i < num_map_points; ++i) {
      const auto id = map_point_ids[i];
      auto map_point_ptr = data_ptr_->getMapPoint(id);
      CHECK(map_point_ptr != nullptr);
      const size_t start_ind = 2 + i * (1 + 2 * MapPointDual::getSize()) +
                               num_frames * (1 + 2 * FrameDual::getSize());
      data_out[start_ind] = static_cast<double>(id);
      // Averaged data
      for (size_t j = 0; j < map_point_ptr->average_state_.getSize(); ++j) {
        data_out[start_ind + 1 + j] = map_point_ptr->average_state_[j];
      }
      // Dual variables
      MapPointDual dual(id);
      CHECK(map_point_ptr->getDualData(neigh_id, dual));
      for (size_t j = 0; j < dual.getSize(); ++j) {
        data_out[start_ind + 1 + dual.getSize() + j] = dual[j];
      }
    }

    MPI_Send(data_out.data(), data_out.size(), MPI_DOUBLE, neigh_id + 1,
             neigh_id + 1 + tagoffset_average_to_worker, MPI_COMM_WORLD);
  }
}

AsyncWorkerNode::AsyncWorkerNode(const int num_workers, const int rank,
                                 DataSharedPtr data_ptr)
    : num_workers_(num_workers),
      rank_(rank),
      data_ptr_(data_ptr),
      local_iterations_(0),
      global_iterations_(0),
      last_saved_state_(-1) {
  CHECK(data_ptr_ != nullptr);
  data_ptr_->writeOutResult("cent_async_" + std::to_string(global_iterations_));
  optimization_ptr_ = std::make_unique<Optimization>(
      data_ptr_, Optimization::ConsensusType::kCentral);
  CHECK(optimization_ptr_->setupProblem());
  this->startNode();
}

AsyncWorkerNode::~AsyncWorkerNode() {
  process_thread_.join();
  counter_thread_.join();
}

auto AsyncWorkerNode::startNode() -> void {
  int command = -1;
  MPI_Recv(&command, 1, MPI_INT, 0, tagoffset_start + rank_, MPI_COMM_WORLD,
           MPI_STATUS_IGNORE);
  process_thread_ = std::thread(&AsyncWorkerNode::processLoop, this);
  counter_thread_ = std::thread(&AsyncWorkerNode::globalCounter, this);
}

auto AsyncWorkerNode::processLoop() -> void {
  const int num_iterations =
      FLAGS_num_admm_iter * num_workers_ / FLAGS_num_async;
  // Create the timing file
  const std::string timing_filename = FLAGS_result_folder + "/Graph_" +
                                      std::to_string(data_ptr_->getGraphId()) +
                                      "/timing_central_async.csv";
  std::ofstream timing_file;
  timing_file.open(timing_filename);
  const auto time_start = std::chrono::high_resolution_clock::now();
  while (global_iterations_ < num_iterations) {
    auto time1 = std::chrono::high_resolution_clock::now();

    // Optimize
    optimization_ptr_->performOptimization();
    if (global_iterations_ % FLAGS_write_out_iter == 0 &&
        last_saved_state_ != global_iterations_) {
      last_saved_state_ = global_iterations_;
      data_ptr_->writeOutResult("cent_async_" +
                                std::to_string(last_saved_state_));
    }

    // Communicate the data to the master for averaging
    this->communicateData();
    auto time2 = std::chrono::high_resolution_clock::now();
    timing_file << std::chrono::duration_cast<std::chrono::milliseconds>(
                       time1 - time_start)
                       .count()
                << ","
                << std::chrono::duration_cast<std::chrono::milliseconds>(
                       time2 - time_start)
                       .count()
                << std::endl;
    ++local_iterations_;
  }
  timing_file.close();
  const auto& residuals = optimization_ptr_->computeErrors();
  std::string filename = FLAGS_data_folder + "/Graph_" +
                         std::to_string(rank_ - 1) + "/residuals.csv";
  std::ofstream file;
  file.open(filename);
  for (const auto r : residuals) {
    file << r << std::endl;
  }
  file.close();
}

auto AsyncWorkerNode::communicateData() -> void {
  const int num_iterations =
      FLAGS_num_admm_iter * num_workers_ / FLAGS_num_async;

  // Prepare the data for communication to the master node
  const auto& frame_ids = data_ptr_->getCommFrames(this->rank_ - 1);
  const size_t num_frames = frame_ids.size();
  const auto& map_point_ids = data_ptr_->getCommMapPoints(this->rank_ - 1);
  const size_t num_map_points = map_point_ids.size();

  // Prepare the communication data with the following ordering in the vector
  // [num_frames, num_map_points, num_frames * [frame_id, p_x, p_y, p_z, dq_x,
  // dq_y, dq_z, intr, dist], num_map_points * [map_point_id, p_x, p_y, p_z]]
  const size_t num_vars = 2 + num_frames * (1 + FrameDual::getSize()) +
                          num_map_points * (1 + MapPointDual::getSize());
  std::vector<double> communication_data(num_vars);
  communication_data[0] = static_cast<double>(num_frames);
  communication_data[1] = static_cast<double>(num_map_points);

  // Fill in the frame data
  for (size_t i = 0; i < num_frames; ++i) {
    const auto start_ind = 2 + i * (FrameDual::getSize() + 1);
    const auto id = frame_ids[i];
    communication_data[start_ind] = static_cast<double>(id);
    auto frame_ptr = data_ptr_->getFrame(id);
    CHECK(frame_ptr != nullptr);
    // If this frame is classified as an outlier by the worker, communicate this
    // to the master. This is done by sending a negative value at the id
    // position and the frame id at the subsequent slot.
    if (!frame_ptr->is_valid_) {
      LOG(FATAL) << "This should not happen at the moment";
      communication_data[start_ind] = -1;
      communication_data[start_ind + 1] = static_cast<double>(id);
      continue;
    }
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
    const auto start_ind = 2 + num_frames * (FrameDual::getSize() + 1) +
                           i * (MapPointDual::getSize() + 1);
    const auto id = map_point_ids[i];
    communication_data[start_ind] = static_cast<double>(id);
    auto map_point_ptr = data_ptr_->getMapPoint(id);
    CHECK(map_point_ptr != nullptr);
    // If this map_point is classified as an outlier by the worker, communicate
    // this to the master. This is done by sending a negative value at the id
    // position and the map point id at the subsequent slot.
    if (!map_point_ptr->is_valid_) {
      LOG(FATAL) << "This should not happen at the moment";
      communication_data[start_ind] = -1;
      communication_data[start_ind + 1] = static_cast<double>(id);
      continue;
    }
    for (size_t j = 0; j < MapPointDual::getSize(); ++j) {
      communication_data[start_ind + 1 + j] = map_point_ptr->position_[j];
    }
  }

  // Send the data
  MPI_Request request;
  auto val = MPI_Isend(communication_data.data(), communication_data.size(),
                       MPI_DOUBLE, 0, rank_ + tagoffset_state_to_master,
                       MPI_COMM_WORLD, &request);
  CHECK(val == MPI_SUCCESS);
  int flag = 0;
  MPI_Status status;
  MPI_Test(&request, &flag, &status);
  while (global_iterations_ < num_iterations && flag == 0) {
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    MPI_Test(&request, &flag, &status);
  }
  if (flag == 0) {
    MPI_Cancel(&request);
    MPI_Request_free(&request);
    return;
  }
  communication_data.clear();

  // Receive and extract the averaged result
  // The data has the format
  // [num_frames, num_map_points, num_frames * [frame_id, avg(trans, rot,
  // intr, dist), dual(trans, rot, intr, dist)], num_map_points *
  // [map_point_id, avg(pos), dual(pos)]]
  MPI_Request request2;
  const size_t out_vars = 2 + num_frames * (2 * FrameDual::getSize() + 1) +
                          num_map_points * (2 * MapPointDual::getSize() + 1);
  communication_data.resize(out_vars);
  val = MPI_Irecv(communication_data.data(), communication_data.size(),
                  MPI_DOUBLE, 0, rank_ + tagoffset_average_to_worker,
                  MPI_COMM_WORLD, &request2);
  CHECK(val == MPI_SUCCESS);
  flag = 0;
  MPI_Status status2;
  MPI_Test(&request2, &flag, &status2);
  while (global_iterations_ < num_iterations && flag == 0) {
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    MPI_Test(&request2, &flag, &status2);
  }
  if (flag == 0) {
    MPI_Cancel(&request2);
    MPI_Request_free(&request2);
    return;
  }
  CHECK_EQ(static_cast<uint64_t>(communication_data[0]), num_frames);
  CHECK_EQ(static_cast<uint64_t>(communication_data[1]), num_map_points);

  std::unordered_map<uint64_t, FrameDual> incoming_frame_vars;
  for (size_t i = 0; i < num_frames; ++i) {
    FrameDual tmp_state, tmp_dual;
    const auto start_ind = 2 + i * (FrameDual::getSize() * 2 + 1);
    // Check whether this frame is classified as outlier. This can be the case
    // if it was classified by this node or any of the neighboring nodes in the
    // last iteration.
    const double recv_id = communication_data[start_ind] + 1e-8;
    if (recv_id < 0.0) {
      LOG(FATAL) << "This should not happen at the moment (id: " << recv_id
                 << ")";
      const uint64_t id =
          static_cast<uint64_t>(communication_data[start_ind + 1] + 1e-8);
      auto frame_ptr = data_ptr_->getFrame(id);
      CHECK(frame_ptr != nullptr);
      frame_ptr->is_valid_ = false;
      continue;
    }
    const uint64_t id = static_cast<uint64_t>(recv_id);
    auto frame_ptr = data_ptr_->getFrame(id);
    CHECK(frame_ptr != nullptr);
    const double sigma_translation = frame_ptr->sigma_trans_;
    const double sigma_rotation = frame_ptr->sigma_rot_;
    const double sigma_intrinsics = frame_ptr->sigma_intr_;
    const double sigma_distortion = frame_ptr->sigma_dist_;
    for (size_t j = 0; j < FrameDual::getSize(); ++j) {
      tmp_state[j] = communication_data[start_ind + 1 + j];
      frame_ptr->average_state_[j] = tmp_state[j];
      tmp_dual[j] =
          communication_data[start_ind + 1 + FrameDual::getSize() + j];
    }
    incoming_frame_vars[id] = tmp_state;

    // In contrast to the formulation from Zhan et al., here the consensus error
    // term is defined as err = x - (bar(x) - sigma^-1 * dual). In order to use
    // the same consensus error term, here we already scale the dual variable
    // accordingly.
    for (size_t j = 0; j < tmp_dual.getSize(); ++j) {
      if (j < 3) {
        // Translation
        tmp_dual[j] /= sigma_translation;
      } else if (j >= 3 && j < 6) {
        // Rotation
        tmp_dual[j] /= sigma_rotation;
      } else if (j >= 6 && j < 6 + kNumIntrinsicParams) {
        // Intrinsics
        tmp_dual[j] /= sigma_intrinsics;
      } else {
        // Distortion
        tmp_dual[j] /= sigma_distortion;
      }
    }
    frame_ptr->setCentralDual(tmp_dual);
  }
  std::unordered_map<uint64_t, MapPointDual> incoming_map_point_vars;
  for (size_t i = 0; i < num_map_points; ++i) {
    MapPointDual tmp_state, tmp_dual;
    const auto start_ind = 2 + num_frames * (FrameDual::getSize() * 2 + 1) +
                           i * (MapPointDual::getSize() * 2 + 1);
    // Check whether this frame is classified as outlier. This can be the case
    // if it was classified by this node or any of the neighboring nodes in the
    // last iteration.
    const double recv_id = communication_data[start_ind] + 1e-8;
    if (recv_id < 0.0) {
      LOG(FATAL) << "This should not happen at the moment (id: " << recv_id
                 << ")";
      const uint64_t id =
          static_cast<uint64_t>(communication_data[start_ind + 1] + 1e-8);
      auto map_point_ptr = data_ptr_->getMapPoint(id);
      CHECK(map_point_ptr != nullptr);
      map_point_ptr->is_valid_ = false;
      continue;
    }
    const uint64_t id = static_cast<uint64_t>(recv_id);
    auto map_point_ptr = data_ptr_->getMapPoint(id);
    CHECK(map_point_ptr != nullptr);
    const double sigma_map_points = map_point_ptr->sigma_;
    for (size_t j = 0; j < MapPointDual::getSize(); ++j) {
      tmp_state[j] = communication_data[start_ind + 1 + j];
      map_point_ptr->average_state_[j] = tmp_state[j];
      tmp_dual[j] =
          (communication_data[start_ind + 1 + MapPointDual::getSize() + j]) /
          sigma_map_points;
    }
    incoming_map_point_vars[id] = tmp_state;
    map_point_ptr->setCentralDual(tmp_dual);
  }
  CHECK(optimization_ptr_->updateAverages(incoming_frame_vars,
                                          incoming_map_point_vars, false));
}  // namespace dba

auto AsyncWorkerNode::globalCounter() -> void {
  const int num_iterations =
      FLAGS_num_admm_iter * num_workers_ / FLAGS_num_async;
  while (global_iterations_ < num_iterations) {
    int counter = 0;
    MPI_Recv(&counter, 1, MPI_INT, 0, rank_ + tagoffset_counter, MPI_COMM_WORLD,
             MPI_STATUSES_IGNORE);
    global_iterations_ = counter;
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
}

}  // namespace dba
