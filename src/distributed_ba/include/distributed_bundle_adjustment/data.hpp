#pragma once

#include <memory>
#include <mutex>
#include <unordered_map>
#include <unordered_set>

#include "distributed_bundle_adjustment/frame.hpp"
#include "distributed_bundle_adjustment/map_point.hpp"

namespace dba {

class Data {
 public:
  using FrameDual = PoseDual<kNumIntrinsicParams, kNumDistortionParams>;

 public:
  Data() = delete;
  Data(const uint64_t graph_id);
  ~Data();

  auto updateFromOther(const std::shared_ptr<Data> data_ptr) -> bool;

  auto readDataFromFiles(const std::string& path,
                         const std::string& file_extension = "") -> bool;

  auto getMapPointIds() const -> std::vector<uint64_t>;

  auto getFrameIds() const -> std::vector<uint64_t>;

  auto getFrameIds(const uint64_t graph_id) const -> std::vector<uint64_t>;

  auto getMapPoint(const uint64_t id) const -> MapPointSharedPtr;

  auto getCommMapPoints(const uint64_t graph_id) const -> std::vector<uint64_t>;

  auto getGlobalNumberOfMapPoints() const -> size_t {
    return num_map_points_global_;
  }

  auto getFrame(const uint64_t id) const -> FrameSharedPtr;

  auto getCommFrames(const uint64_t graph_id) const -> std::vector<uint64_t>;

  auto getGlobalNumberOfFrames() const -> size_t { return num_frames_global_; }

  auto getGlobalNumberOfObservations() const -> size_t {
    return num_observations_global_;
  }

  auto getGraphId() const -> uint64_t { return graph_id_; }

  auto getNeighbors() const -> std::vector<uint64_t> { return neighbor_ids_; }

  auto writeOutResult(const std::string& ap) -> bool;

 private:
  auto readGlobalProperties() -> bool;

  auto getNumberOfLines(const std::string& filename) -> size_t;

  auto readFrames(const std::string& data_folder,
                  const std::string& appendix = "") -> bool;

  auto readMapPoints(const std::string& data_folder,
                     const std::string& appendix = "") -> bool;

  auto readObservations(const std::string& data_folder) -> bool;

  auto readConnectionGraph(const std::string& data_folder) -> bool;

  auto initializeDualWeights() -> bool;

  const uint64_t graph_id_;
  uint64_t max_frame_id_;
  std::unordered_map<uint64_t, FrameSharedPtr> frames_;
  std::unordered_map<uint64_t, std::vector<uint64_t>> frames_to_comm_;
  std::unordered_map<uint64_t, std::vector<uint64_t>> map_points_to_comm_;
  std::unordered_map<uint64_t, std::vector<uint64_t>> graph_to_frames_;
  std::unordered_map<uint64_t, std::vector<uint64_t>> graph_to_map_points_;
  std::unordered_map<uint64_t, MapPointSharedPtr> map_points_;

  // Store information related to the global problem
  size_t num_frames_global_;
  size_t num_map_points_global_;
  size_t num_observations_global_;

  // Store the neighbor ids for easy access (ordered!)
  std::vector<uint64_t> neighbor_ids_;

  // Mutex to handle copy and writing out internally
  std::mutex write_mutex_;
};

using DataSharedPtr = std::shared_ptr<Data>;
using DataUniquePtr = std::shared_ptr<Data>;

}  // namespace dba
