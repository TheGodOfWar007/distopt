#include <gflags/gflags.h>
#include <chrono>
#include <experimental/filesystem>
#include <fstream>
#include <iomanip>
#include <queue>

#include "distributed_bundle_adjustment/asynchronous_communication.hpp"
#include "distributed_bundle_adjustment/common.hpp"

namespace filesystem = std::experimental::filesystem;

DECLARE_uint64(num_admm_iter);
DECLARE_uint32(write_out_iter);
DECLARE_string(result_folder);

constexpr double id_eps_ = 1e-9;
constexpr size_t key_offset_multiplier = 10000;

namespace dba {

AsynchronousCoordinator::AsynchronousCoordinator(
    const std::vector<DataSharedPtr>& data)
    : iteration_counter_(0) {
  CHECK(!data.empty());
  for (const auto& local_ptr : data) {
    CHECK(local_ptr != nullptr);
    const auto id = local_ptr->getGraphId();
    data_map_[id] = local_ptr;
  }

  // Create a world group in order to enable the construction of the pairwise
  // communicators with the neighbors later.
  MPI_Comm_size(MPI_COMM_WORLD, &world_size_);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  CHECK_EQ(rank, 0);
  MPI_Group group_world;
  MPI_Comm_group(MPI_COMM_WORLD, &group_world);
  const auto num_nodes = world_size_ - 1;
  constexpr int memory_offset_this = 0;

  // Create the window for storing the global counts of the iterations
  MPI_Win_allocate(num_nodes * sizeof(int), sizeof(int), MPI_INFO_NULL,
                   MPI_COMM_WORLD, &counter_storage_, &counter_window_);
  std::vector<int> counter_init(num_nodes, 0);
  MPI_Win_lock(MPI_LOCK_EXCLUSIVE, rank, 0, counter_window_);
  MPI_Accumulate(counter_init.data(), num_nodes, MPI_INT, rank,
                 memory_offset_this, num_nodes, MPI_INT, MPI_REPLACE,
                 counter_window_);
  MPI_Win_unlock(0, counter_window_);

  // Create the window for storing the indicators signaling that data is
  // requested
  MPI_Win_allocate(num_nodes * sizeof(int), sizeof(int), MPI_INFO_NULL,
                   MPI_COMM_WORLD, &flag_storage_, &flag_window_);
  last_flag_state_.resize(num_nodes, 0);
  MPI_Win_lock(MPI_LOCK_EXCLUSIVE, rank, 0, flag_window_);
  MPI_Accumulate(last_flag_state_.data(), num_nodes, MPI_INT, rank,
                 memory_offset_this, num_nodes, MPI_INT, MPI_REPLACE,
                 flag_window_);
  MPI_Win_unlock(0, flag_window_);

  // Create the finish flag window
  MPI_Win_allocate(num_nodes * sizeof(int), sizeof(int), MPI_INFO_NULL,
                   MPI_COMM_WORLD, &finish_flags_, &finish_flag_window_);
  MPI_Win_lock(MPI_LOCK_EXCLUSIVE, rank, 0, finish_flag_window_);
  std::vector<int> finish_flags(num_nodes, 0);
  MPI_Accumulate(finish_flags.data(), num_nodes, MPI_INT, rank,
                 memory_offset_this, num_nodes, MPI_INT, MPI_REPLACE,
                 finish_flag_window_);
  MPI_Win_unlock(0, finish_flag_window_);

  // Construct the groups for the different communicators
  std::map<uint64_t, MPI_Group> comm_groups;
  for (const auto& local_data_ptr : data) {
    const auto local_id = local_data_ptr->getGraphId();
    const auto n_ids = local_data_ptr->getNeighbors();
    for (const auto n_id : n_ids) {
      if (n_id == local_id) {
        continue;
      }
      const auto key = computeKeyFromIds(local_id, n_id);
      if (comm_groups.count(key)) {
        // We allready created this group
        continue;
      }
      // Create the corresponding group for this pair
      // Since the ranks in the communicators are always starting at 0, we
      // define the logic here, that the node with the lower absolute rank has
      // rank zero, the other rank 1.
      std::vector<int> global_ranks(3);
      if (n_id < local_id) {
        global_ranks[0] = 0;
        global_ranks[1] = static_cast<int>(n_id + 1);
        global_ranks[2] = static_cast<int>(local_id + 1);
      } else {
        global_ranks[0] = 0;
        global_ranks[1] = static_cast<int>(local_id + 1);
        global_ranks[2] = static_cast<int>(n_id + 1);
      }

      MPI_Group tmp_group;
      MPI_Group_incl(group_world, global_ranks.size(), global_ranks.data(),
                     &tmp_group);
      comm_groups[key] = tmp_group;
    }
  }

  // Construct the actual Communicators and the windows
  for (const auto& [key, group] : comm_groups) {
    const auto [id_lo, id_hi] = computeIdsFromKey(key);
    // Create the communicator
    MPI_Comm tmp_comm;
    MPI_Comm_create_group(MPI_COMM_WORLD, group, static_cast<int>(key),
                          &tmp_comm);
    const EdgeId edge_ab({id_lo, id_hi});
    const EdgeId edge_ba({id_hi, id_lo});
    communicator_map_[edge_ab] = tmp_comm;
    communicator_map_[edge_ba] = tmp_comm;

    data_buffer_map_[edge_ab] = std::vector<double>();
    data_buffer_map_[edge_ba] = std::vector<double>();
  }

  initializeBuffers();

  main_thread_ = std::thread(&AsynchronousCoordinator::mainThread, this);
}

AsynchronousCoordinator::~AsynchronousCoordinator() {
  main_thread_.join();
  MPI_Win_free(&counter_window_);
  MPI_Win_free(&flag_window_);
}

auto AsynchronousCoordinator::initializeBuffers() -> void {
  for (auto& [edge, data_storage] : data_buffer_map_) {
    const auto& [id_this, id_neigh] = edge;
    CHECK(data_map_.count(id_this));
    auto local_data_ptr = data_map_[id_this];
    CHECK(local_data_ptr != nullptr);
    const auto& frame_ids = local_data_ptr->getCommFrames(id_neigh);
    const size_t num_frames = frame_ids.size();
    const auto& map_point_ids = local_data_ptr->getCommMapPoints(id_neigh);
    const size_t num_map_points = map_point_ids.size();
    CHECK(num_frames > 0 || num_map_points > 0);

    // Just to be save we store also the id in the array, resulting in the
    // following form:
    // num_frames, num_map_points, num_frames * [id, dual(p_x, p_y, p_z, dq_x,
    // dq_y, dq_z, intrinsics, distortion)], num_map_points * [id, dual(p_x,
    // p_y, p_z)]]
    const size_t num_vars = 2 + num_frames * (FrameDual::getSize() + 1) +
                            num_map_points * (MapPointDual::getSize() + 1);
    data_storage.resize(num_vars, 0);

    // Initialize the actual values
    data_storage[0] = static_cast<double>(num_frames);
    data_storage[1] = static_cast<double>(num_map_points);
    for (size_t i = 0; i < num_frames; ++i) {
      const auto id = frame_ids[i];
      auto frame_ptr = local_data_ptr->getFrame(id);
      CHECK(frame_ptr != nullptr);
      const size_t start_ind = 2 + i * (FrameDual::getSize() + 1);
      if (!frame_ptr->is_valid_) {
        data_storage[start_ind] = -1.0;
        data_storage[start_ind + 1] = static_cast<double>(id);
        continue;
      }
      data_storage[start_ind] = static_cast<double>(id);
      FrameDual dual(id);
      CHECK(frame_ptr->getDualData(id_neigh, dual));
      for (size_t j = 0; j < dual.getSize(); ++j) {
        data_storage[start_ind + 1 + j] = dual[j];
      }
    }
    auto time3 = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < num_map_points; ++i) {
      const auto id = map_point_ids[i];
      auto map_point_ptr = local_data_ptr->getMapPoint(id);
      CHECK(map_point_ptr != nullptr);
      const size_t start_ind = 2 + num_frames * (FrameDual::getSize() + 1) +
                               i * (MapPointDual::getSize() + 1);
      if (!map_point_ptr->is_valid_) {
        data_storage[start_ind] = -1.0;
        data_storage[start_ind + 1] = static_cast<double>(id);
        continue;
      }
      data_storage[start_ind] = static_cast<double>(id);
      MapPointDual dual(id);
      CHECK(map_point_ptr->getDualData(id_neigh, dual));
      for (size_t j = 0; j < dual.getSize(); ++j) {
        data_storage[start_ind + 1 + j] = dual[j];
      }
    }
  }
}

auto AsynchronousCoordinator::mainThread() -> void {
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  constexpr int target_rank = 0;
  constexpr int memory_offset_read = 0;
  const int num_counter_slots = world_size - 1;
  MPI_Barrier(MPI_COMM_WORLD);
  std::vector<int> counter(num_counter_slots, 0);
  std::string iteration_timer_filename =
      FLAGS_result_folder + "/iteration_timing_decent_async.csv";
  std::ofstream iteration_timer_file;
  iteration_timer_file.open(iteration_timer_filename);
  iteration_timer_file.close();
  const auto start_time = std::chrono::high_resolution_clock::now();
  const std::string debug_time_filename =
      FLAGS_result_folder + "/debug_timing.csv";
  std::ofstream debug_timer_file;
  debug_timer_file.open(debug_time_filename);
  bool run_loop = true;
  while (run_loop) {
    const auto time1 = std::chrono::high_resolution_clock::now();
    if (updateCounter()) {
      const auto curr_time = std::chrono::high_resolution_clock::now();
      iteration_timer_file.open(iteration_timer_filename, std::ios_base::app);
      iteration_timer_file
          << iteration_counter_ << ","
          << std::chrono::duration_cast<std::chrono::milliseconds>(curr_time -
                                                                   start_time)
                 .count()
          << std::endl;
      iteration_timer_file.close();
      LOG(INFO) << "Global iteration " << iteration_counter_ << " of "
                << FLAGS_num_admm_iter;
    }

    // Update the buffers (received duals)
    const auto time2 = std::chrono::high_resolution_clock::now();
    updateBuffers();

    // Check for any requested updates
    const auto time3 = std::chrono::high_resolution_clock::now();
    std::vector<int> update_check(num_counter_slots, 0);
    MPI_Win_lock(MPI_LOCK_SHARED, target_rank, 0, flag_window_);
    auto res = MPI_Get_accumulate(flag_storage_, num_counter_slots, MPI_INT,
                                  update_check.data(), num_counter_slots,
                                  MPI_INT, target_rank, 0, num_counter_slots,
                                  MPI_INT, MPI_NO_OP, flag_window_);
    MPI_Win_unlock(target_rank, flag_window_);
    const auto time4 = std::chrono::high_resolution_clock::now();
    for (uint64_t i = 0; i < num_counter_slots; ++i) {
      CHECK_GE(update_check[i], last_flag_state_[i]);
      if (update_check[i] > last_flag_state_[i]) {
        sendUpdates(i);
        last_flag_state_[i] = update_check[i];
      }
    }
    const auto time5 = std::chrono::high_resolution_clock::now();

    // See whether the nodes are finished
    std::vector<int> finish_flags(num_counter_slots, 0);
    MPI_Win_lock(MPI_LOCK_SHARED, target_rank, 0, finish_flag_window_);
    res = MPI_Get_accumulate(finish_flags_, num_counter_slots, MPI_INT,
                             finish_flags.data(), num_counter_slots, MPI_INT,
                             target_rank, memory_offset_read, num_counter_slots,
                             MPI_INT, MPI_NO_OP, finish_flag_window_);
    MPI_Win_unlock(target_rank, finish_flag_window_);
    auto min_itr = std::min_element(finish_flags.begin(), finish_flags.end());
    CHECK(min_itr != finish_flags.end());
    const int min_count = (*min_itr);
    const auto time6 = std::chrono::high_resolution_clock::now();
    auto dt1 =
        std::chrono::duration_cast<std::chrono::microseconds>(time2 - time1)
            .count();
    auto dt2 =
        std::chrono::duration_cast<std::chrono::microseconds>(time3 - time2)
            .count();
    auto dt3 =
        std::chrono::duration_cast<std::chrono::microseconds>(time4 - time3)
            .count();
    auto dt4 =
        std::chrono::duration_cast<std::chrono::microseconds>(time5 - time4)
            .count();
    auto dt5 =
        std::chrono::duration_cast<std::chrono::microseconds>(time6 - time5)
            .count();
    debug_timer_file << dt1 << "," << dt2 << "," << dt3 << "," << dt4 << ","
                     << dt5 << std::endl;
    run_loop = (min_count < 1) || (iteration_counter_ < FLAGS_num_admm_iter);
  }
  debug_timer_file.close();
}

auto AsynchronousCoordinator::updateCounter() -> bool {
  constexpr int target_rank = 0;
  constexpr int memory_offset_read = 0;
  const int num_counter_slots = world_size_ - 1;
  std::vector<int> counter_data(num_counter_slots);
  MPI_Win_lock(MPI_LOCK_SHARED, target_rank, 0, counter_window_);
  auto res = MPI_Get_accumulate(
      counter_storage_, num_counter_slots, MPI_INT, counter_data.data(),
      num_counter_slots, MPI_INT, target_rank, memory_offset_read,
      num_counter_slots, MPI_INT, MPI_NO_OP, counter_window_);
  MPI_Win_unlock(target_rank, counter_window_);
  auto min_itr = std::min_element(counter_data.begin(), counter_data.end());
  CHECK(min_itr != counter_data.end());
  const int min_count = (*min_itr);
  bool updated_counter = (iteration_counter_ < min_count);
  iteration_counter_ = min_count;
  return updated_counter;
}

auto AsynchronousCoordinator::updateBuffers() -> void {
  for (auto& [id_pair, data_buffer] : data_buffer_map_) {
    checkAndUpdateBuffer(id_pair, data_buffer);
  }
}

auto AsynchronousCoordinator::sendUpdates(const uint64_t& node_id) -> void {
  auto data_ptr = data_map_[node_id];
  CHECK(data_ptr != nullptr);
  CHECK_EQ(data_ptr->getGraphId(), node_id);
  const auto& n_ids = data_ptr->getNeighbors();
  std::vector<MPI_Request> requests;
  for (const auto& n_id : n_ids) {
    if (n_id == node_id) {
      continue;
    }
    // Communication edges are defined as <from, to>, hence we want all the
    // nodes n_i for <n_i, node_id>.
    EdgeId tmp_id({n_id, node_id});
    auto& dual_data = data_buffer_map_[tmp_id];
    const int target_process = (node_id < n_id) ? 1 : 2;
    const int message_tag =
        static_cast<int>(computeDirectedKeyFromIds(n_id, node_id));
    MPI_Request tmp_request;
    MPI_Isend(dual_data.data(), dual_data.size(), MPI_DOUBLE, target_process,
              message_tag, communicator_map_[tmp_id], &tmp_request);
    requests.push_back(tmp_request);
  }

  for (size_t i = 0; i < requests.size(); ++i) {
    MPI_Wait(&requests[i], MPI_STATUS_IGNORE);
  }
}

auto AsynchronousCoordinator::checkAndUpdateBuffer(const EdgeId& id,
                                                   std::vector<double>& buffer)
    -> bool {
  // An edge is defined by the communication direction id.first to id.second,
  // i.e. id.first is the sender and id.second is the receiver.
  auto& communicator = communicator_map_[id];
  const int source_rank = (id.first < id.second) ? 1 : 2;
  const int message_tag =
      static_cast<int>(computeDirectedKeyFromIds(id.first, id.second));
  int message_flag = 0;
  auto val = MPI_Iprobe(source_rank, message_tag, communicator, &message_flag,
                        MPI_STATUS_IGNORE);
  if (message_flag == 0) {
    return false;
  }

  MPI_Recv(buffer.data(), buffer.size(), MPI_DOUBLE, source_rank, message_tag,
           communicator, MPI_STATUS_IGNORE);
  return true;
}

AsynchronousCommunication::AsynchronousCommunication(
    DataSharedPtr data_ptr, DataSharedPtr data_copy_ptr)
    : data_ptr_(data_ptr),
      data_copy_ptr_(data_copy_ptr),
      run_(false),
      stop_(false),
      has_outgoing_data_(false),
      iteration_counter_(0),
      global_counter_(0) {
  CHECK(data_ptr != nullptr);
  CHECK(data_ptr != data_copy_ptr);
  optimization_ptr_ = std::make_unique<Optimization>(
      data_ptr, Optimization::ConsensusType::kDecentral);
  neighbor_ids_ = data_ptr_->getNeighbors();

  // Create a world group in order to enable the construction of the pairwise
  // communicators with the neighbors later.
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  CHECK_EQ(rank, data_ptr_->getGraphId() + 1);
  MPI_Group group_world;
  MPI_Comm_group(MPI_COMM_WORLD, &group_world);
  const auto graph_id = data_ptr->getGraphId();

  // Create the window for storing the global counts of the iterations
  const int num_slots = world_size - 1;
  MPI_Win_allocate(num_slots * sizeof(int), sizeof(int), MPI_INFO_NULL,
                   MPI_COMM_WORLD, &counter_storage_, &counter_window_);

  // Create the window for the tag communication
  MPI_Win_allocate(num_slots * sizeof(int), sizeof(int), MPI_INFO_NULL,
                   MPI_COMM_WORLD, &flag_storage_, &flag_window_);

  MPI_Win_allocate(0, sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD,
                   &finish_flags_, &finish_flag_window_);

  // In order to allow a global order for the communicator generation, we create
  // a lexicographical index using the neighbor id and the rank. This allows us
  // to order the combination and make sure that we avoid running into a lock.
  std::map<uint64_t, MPI_Group> comm_groups;

  for (const auto& n_id : neighbor_ids_) {
    if (n_id == graph_id) {
      continue;
    }

    // Create the corresponding group for the neighbor
    // Since the ranks in the communicators are always starting at 0, we
    // define the logic here, that the node with the lower absolute rank has
    // rank zero, the other rank 1.
    std::vector<int> global_ranks(3);
    if (n_id < graph_id) {
      global_ranks[0] = 0;
      global_ranks[1] = static_cast<int>(n_id + 1);
      global_ranks[2] = rank;
    } else {
      global_ranks[0] = 0;
      global_ranks[1] = rank;
      global_ranks[2] = static_cast<int>(n_id + 1);
    }
    MPI_Group tmp_group;
    MPI_Group_incl(group_world, global_ranks.size(), global_ranks.data(),
                   &tmp_group);

    // Create the lexicographical index
    const auto lex_ind = computeKeyFromIds(graph_id, n_id);

    comm_groups[lex_ind] = tmp_group;
  }

  // Create the actual communicators as well as the corresponding windows
  for (auto& [key, group] : comm_groups) {
    const auto [id_lo, id_hi] = computeIdsFromKey(key);
    const auto n_id = (id_lo == graph_id) ? id_hi : id_lo;

    // Create the communicator
    MPI_Comm tmp_comm;
    MPI_Comm_create_group(MPI_COMM_WORLD, group, static_cast<int>(key),
                          &tmp_comm);
    communicator_map_[n_id] = tmp_comm;

    // We need to allocate the local space for the window to hold the incoming
    // dual data from the neighbor.
    const auto& frame_ids = data_ptr_->getCommFrames(n_id);
    const size_t num_frames = frame_ids.size();
    const auto& map_point_ids = data_ptr_->getCommMapPoints(n_id);
    const size_t num_map_points = map_point_ids.size();
    CHECK(num_frames > 0 || num_map_points > 0);
    const size_t num_vars = 2 + num_frames * (FrameDual::getSize() + 1) +
                            num_map_points * (MapPointDual::getSize() + 1);
    outgoing_buffer_[n_id] = std::vector<double>(num_vars);
  }

  // Setup the base optimization problem
  optimization_ptr_->setupProblem();

  main_thread_ = std::thread(&AsynchronousCommunication::mainThread, this);
}

AsynchronousCommunication::~AsynchronousCommunication() {
  main_thread_.join();

  // Free the global windows
  MPI_Win_free(&counter_window_);
  MPI_Win_free(&flag_window_);
}

auto AsynchronousCommunication::mainThread() -> void {
  // Block until the start command (from Node 0) is received
  MPI_Barrier(MPI_COMM_WORLD);

  // Create the timing file
  const std::string data_folder =
      FLAGS_result_folder + "/Graph_" + std::to_string(data_ptr_->getGraphId());
  if (!filesystem::is_directory(data_folder)) {
    CHECK(filesystem::create_directory(data_folder));
  }

  std::ofstream timing_file;
  const std::string timing_filename = data_folder + "/timing_async.csv";

  //  std::ofstream timing_file;
  timing_file.open(timing_filename);
  auto time_start = std::chrono::high_resolution_clock::now();
  while (global_counter_ < FLAGS_num_admm_iter) {
    auto time1 = std::chrono::high_resolution_clock::now();
    updateReceivedDuals();
    auto time2 = std::chrono::high_resolution_clock::now();
    optimization_ptr_->performOptimization(); // This is equation 8, This solves all nodes of the local graph of an agent.
    auto time3 = std::chrono::high_resolution_clock::now();
    updateDuals(); // Equation 9 and 10 (and 11 too)
    auto time4 = std::chrono::high_resolution_clock::now();
    communicateData();
    auto time5 = std::chrono::high_resolution_clock::now();
    optimization_ptr_->updateDuals();
    auto time6 = std::chrono::high_resolution_clock::now();
    writeOutStatus();
    auto time7 = std::chrono::high_resolution_clock::now();
    communicateData();
    auto time8 = std::chrono::high_resolution_clock::now();
    auto dt1 =
        std::chrono::duration_cast<std::chrono::milliseconds>(time2 - time1)
            .count();
    auto dt2 =
        std::chrono::duration_cast<std::chrono::milliseconds>(time3 - time2)
            .count();
    auto dt3 =
        std::chrono::duration_cast<std::chrono::milliseconds>(time4 - time3)
            .count();
    auto dt4 =
        std::chrono::duration_cast<std::chrono::milliseconds>(time5 - time4)
            .count();
    auto dt5 =
        std::chrono::duration_cast<std::chrono::milliseconds>(time6 - time5)
            .count();
    auto dt6 =
        std::chrono::duration_cast<std::chrono::milliseconds>(time7 - time6)
            .count();
    auto dt7 =
        std::chrono::duration_cast<std::chrono::milliseconds>(time8 - time7)
            .count();
    timing_file << dt1 << "," << dt2 << "," << dt3 << "," << dt4 << "," << dt5
                << "," << dt6 << "," << dt7 << std::endl;
    ++iteration_counter_;
  }
  timing_file.close();

  const int local_rank_other = 0;  // Central Communicator
  const int memory_offset_this = static_cast<int>(data_ptr_->getGraphId());
  MPI_Win_lock(MPI_LOCK_EXCLUSIVE, local_rank_other, 0, finish_flag_window_);
  int write_num = 1;
  int dummy_addr = 0;
  MPI_Fetch_and_op(&write_num, &dummy_addr, MPI_INT, local_rank_other,
                   memory_offset_this, MPI_SUM, finish_flag_window_);
  MPI_Win_unlock(local_rank_other, finish_flag_window_);
}

auto AsynchronousCommunication::communicateData() -> void {
  const size_t num_neighbors = neighbor_ids_.size() - 1;
  if (!has_outgoing_data_) {
    outgoing_comm_.resize(num_neighbors);
    size_t data_ind = 0;

    for (const auto& n_id : neighbor_ids_) {
      if (n_id == data_ptr_->getGraphId()) {
        continue;
      }
      const int local_rank_this = (data_ptr_->getGraphId() < n_id) ? 1 : 2;
      const int local_rank_other = 0;  // Central communicator
      const auto& frame_ids = data_ptr_->getCommFrames(n_id);
      const size_t num_frames = frame_ids.size();
      const auto& map_point_ids = data_ptr_->getCommMapPoints(n_id);
      const size_t num_map_points = map_point_ids.size();
      CHECK(num_frames > 0 || num_map_points > 0);

      // Prepare the local array to be written to the neihbors windows
      CHECK(outgoing_buffer_.count(n_id));
      auto& data_storage = outgoing_buffer_[n_id];
      data_storage[0] = static_cast<double>(num_frames);
      data_storage[1] = static_cast<double>(num_map_points);
      for (size_t i = 0; i < num_frames; ++i) {
        const auto id = frame_ids[i];
        auto frame_ptr = data_ptr_->getFrame(id);
        CHECK(frame_ptr != nullptr);
        const size_t start_ind = 2 + i * (FrameDual::getSize() + 1);
        if (!frame_ptr->is_valid_) {
          data_storage[start_ind] = -1.0;
          data_storage[start_ind + 1] = static_cast<double>(id);
          continue;
        }
        data_storage[start_ind] = static_cast<double>(id);
        FrameDual dual(id);
        CHECK(frame_ptr->getDualData(n_id, dual));
        for (size_t j = 0; j < dual.getSize(); ++j) {
          data_storage[start_ind + 1 + j] = dual[j];
        }
      }

      for (size_t i = 0; i < num_map_points; ++i) {
        const auto id = map_point_ids[i];
        auto map_point_ptr = data_ptr_->getMapPoint(id);
        CHECK(map_point_ptr != nullptr);
        const size_t start_ind = 2 + num_frames * (FrameDual::getSize() + 1) +
                                 i * (MapPointDual::getSize() + 1);
        if (!map_point_ptr->is_valid_) {
          data_storage[start_ind] = -1.0;
          data_storage[start_ind + 1] = static_cast<double>(id);
          continue;
        }
        data_storage[start_ind] = static_cast<double>(id);
        MapPointDual dual(id);
        CHECK(map_point_ptr->getDualData(n_id, dual));
        for (size_t j = 0; j < dual.getSize(); ++j) {
          data_storage[start_ind + 1 + j] = dual[j];
        }
      }

      // Write the data to the shared storage
      CHECK(communicator_map_.count(n_id));
      const int message_tag = static_cast<int>(
          computeDirectedKeyFromIds(data_ptr_->getGraphId(), n_id));
      MPI_Isend(data_storage.data(), data_storage.size(), MPI_DOUBLE,
                local_rank_other, message_tag, communicator_map_[n_id],
                &outgoing_comm_[data_ind]);

      // Increment the index for the data pointers...
      ++data_ind;
    }
    has_outgoing_data_ = true;
  } else {
    // Wait for the sending completion
    for (size_t j = 0; j < num_neighbors; ++j) {
      MPI_Status status;
      MPI_Wait(&outgoing_comm_[j], &status);
    }
    outgoing_comm_.clear();
    has_outgoing_data_ = false;
  }
}

auto AsynchronousCommunication::updateReceivedDuals() -> void {
  // Signal the coordinator that we want to receive the updates
  const int local_rank_other = 0;  // Central Communicator
  const int memory_offset_this = static_cast<int>(data_ptr_->getGraphId());
  MPI_Win_lock(MPI_LOCK_EXCLUSIVE, local_rank_other, 0, flag_window_);
  int write_num = 1;
  int dummy_addr = 0;
  MPI_Fetch_and_op(&write_num, &dummy_addr, MPI_INT, local_rank_other,
                   memory_offset_this, MPI_SUM, flag_window_);
  MPI_Win_unlock(local_rank_other, flag_window_);

  for (const auto& n_id : neighbor_ids_) {
    if (n_id == data_ptr_->getGraphId()) continue;

    const int local_rank_this = (data_ptr_->getGraphId() < n_id) ? 1 : 2;
    const auto& frame_ids = data_ptr_->getCommFrames(n_id);
    const size_t num_frames = frame_ids.size();
    const auto& map_point_ids = data_ptr_->getCommMapPoints(n_id);
    const size_t num_map_points = map_point_ids.size();
    const size_t num_vars = 2 + num_frames * (FrameDual::getSize() + 1) +
                            num_map_points * (MapPointDual::getSize() + 1);
    CHECK(num_frames > 0 || num_map_points > 0);

    // Receive the shared information from the coordinator
    const int message_tag = static_cast<int>(
        computeDirectedKeyFromIds(n_id, data_ptr_->getGraphId()));
    std::vector<double> data_storage(num_vars);
    MPI_Recv(data_storage.data(), num_vars, MPI_DOUBLE, local_rank_other,
             message_tag, communicator_map_[n_id], MPI_STATUS_IGNORE);

    CHECK_EQ(static_cast<size_t>(data_storage[0] + id_eps_), num_frames);
    CHECK_EQ(static_cast<size_t>(data_storage[1] + id_eps_), num_map_points);

    // Extract the frame data
    for (size_t i = 0; i < num_frames; ++i) {
      const auto start_ind = 2 + i * (FrameDual::getSize() + 1);
      if (data_storage[start_ind] < 0.0) {
        LOG(FATAL) << "this should not happen at the moment";
        // This means that this frame was classified as outlier by the
        // neighbor, drop the neighbor.
        // TODO: implement a suitable strategy to actually handle this
        // case. continue;
      }
      const uint64_t id =
          static_cast<uint64_t>(data_storage[start_ind] + id_eps_);
      FrameDual dual(id);
      for (size_t j = 0; j < FrameDual::getSize(); ++j) {
        dual[j] = data_storage[start_ind + 1 + j];
      }
      auto frame_ptr = data_ptr_->getFrame(id);
      CHECK(frame_ptr != nullptr);
      frame_ptr->setCommDual(n_id, dual);
    }

    // Extract the map point data
    for (size_t i = 0; i < num_map_points; ++i) {
      const auto start_ind = 2 + num_frames * (FrameDual::getSize() + 1) +
                             i * (MapPointDual::getSize() + 1);
      if (data_storage[start_ind] < 0.0) {
        LOG(FATAL) << "This should not happen at the moment";
        // This means that this map point was classified as outlier by the
        // neighbor, drop the neighbor.
        // TODO: implement a suitable strategy to actually handle this
        // case. continue;
      }
      const uint64_t id =
          static_cast<uint64_t>(data_storage[start_ind] + id_eps_);
      MapPointDual dual(id);
      for (size_t j = 0; j < MapPointDual::getSize(); ++j) {
        dual[j] = data_storage[start_ind + 1 + j];
      }
      auto map_point_ptr = data_ptr_->getMapPoint(id);
      CHECK(map_point_ptr != nullptr);
      map_point_ptr->setCommDual(n_id, dual);
    }
  }
}

auto AsynchronousCommunication::updateDuals() -> void {
  const auto& frame_ids = data_ptr_->getFrameIds();
  for (const auto& id : frame_ids) {
    auto frame_ptr = data_ptr_->getFrame(id);
    CHECK(frame_ptr != nullptr);
    frame_ptr->updateDualVariables();
    //    CHECK(frame_ptr->updateDualVariables());
  }
  const auto& map_point_ids = data_ptr_->getMapPointIds();
  for (const auto& id : map_point_ids) {
    auto map_point_ptr = data_ptr_->getMapPoint(id);
    CHECK(map_point_ptr != nullptr);
    map_point_ptr->updateDualVariables();
    //    CHECK(map_point_ptr->updateDualVariables());
  }
}

auto AsynchronousCommunication::writeOutStatus() -> void {
  // Maintain and check the global memory kept by node 0
  const int memory_offset_write = static_cast<int>(data_ptr_->getGraphId());
  constexpr int target_rank = 0;
  MPI_Win_lock(MPI_LOCK_EXCLUSIVE, target_rank, 0, counter_window_);
  constexpr int memory_offset_other = 0;

  // Write the local iteration count at the corresponding memory address
  constexpr int num_send = 1;
  int dummy_addr;
  MPI_Fetch_and_op(&iteration_counter_, &dummy_addr, MPI_INT, target_rank,
                   memory_offset_write, MPI_REPLACE, counter_window_);

  // Get the iteration counts from all agents
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  const int num_timer_slots = world_size - 1;
  std::vector<int> counter(num_timer_slots);
  constexpr int memory_offset_read = 0;
  auto res = MPI_Get_accumulate(
      counter_storage_, num_timer_slots, MPI_INT, counter.data(),
      num_timer_slots, MPI_INT, target_rank, memory_offset_read,
      num_timer_slots, MPI_INT, MPI_NO_OP, counter_window_);
  MPI_Win_unlock(target_rank, counter_window_);

  // Check the counter values and write out the current state if requested
  auto min_itr = std::min_element(counter.begin(), counter.end());
  CHECK(min_itr != counter.end());
  const int min_count = (*min_itr);
  if (min_count > global_counter_) {
    global_counter_ = min_count;
    data_ptr_->writeOutResult("decent_async_" +
                              std::to_string(global_counter_));
  }
}

}  // namespace dba
