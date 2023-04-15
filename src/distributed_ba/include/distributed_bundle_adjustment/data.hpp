#pragma once

#include <memory>
#include <mutex>
#include <unordered_map>
#include <unordered_set>

#include "distributed_bundle_adjustment/frame.hpp"
#include "distributed_bundle_adjustment/map_point.hpp"

/// @brief adding to dba namespace
namespace dba {

/// @brief creating Data class
class Data {
 public:
  /// @brief define FrameDual as PoseDual with kNumIntrinsic Params and kNumDistortionParams
  using FrameDual = PoseDual<kNumIntrinsicParams, kNumDistortionParams>;

 public:
  /// @brief explicitly delete default Data constructor, necessitating inputs
  Data() = delete;
  /// @brief Data class constructor declaration
  /// @param graph_id graph_id
  Data(const uint64_t graph_id);
  /// @brief Data class destructor
  /// @details destroys an object when it is out of scope, deleted, or the program terminates
  ~Data();

  /// @brief update current instance with ptr to other Data instance
  /// @param data_ptr ptr to other data instance
  /// @return bool
  auto updateFromOther(const std::shared_ptr<Data> data_ptr) -> bool;

  /// @brief read data from input path with input file extension
  /// @param path 
  /// @param file_extension 
  /// @return bool
  auto readDataFromFiles(const std::string& path,
                         const std::string& file_extension = "") -> bool;

  /// @brief get map point ids
  /// @return vector of map point ids
  auto getMapPointIds() const -> std::vector<uint64_t>;

  /// @brief get frame ids
  /// @return vector of frame ids
  auto getFrameIds() const -> std::vector<uint64_t>;

  /// @brief get FrameIds associated with input GraphId
  /// @param graph_id input GraphId
  /// @return vector of FrameIds associated with input GraphId
  auto getFrameIds(const uint64_t graph_id) const -> std::vector<uint64_t>;

  /// @brief return pointer to location of MapPoint of desired id
  /// @param id desired MapPoint id
  /// @return shared pointer for desired MapPoint object
  auto getMapPoint(const uint64_t id) const -> MapPointSharedPtr;

  /// @brief return pointer to MapPoints associated with a GraphId
  /// @param graph_id desired GraphId
  /// @return vector of MapPointIds associated with input GraphId
  auto getCommMapPoints(const uint64_t graph_id) const -> std::vector<uint64_t>;

  /// @brief 
  /// @return number of map points in global problem
  auto getGlobalNumberOfMapPoints() const -> size_t {
    return num_map_points_global_;
  }

  /// @brief return pointer to location of Frame with id FrameId
  /// @param id desired Frame id
  /// @return shared pointer to frame with FrameId id
  auto getFrame(const uint64_t id) const -> FrameSharedPtr;

  /// @brief return pointer to FrameIds associated with a GraphId
  /// @param graph_id desired GraphId
  /// @return vector of FrameIds associated with input GraphId
  auto getCommFrames(const uint64_t graph_id) const -> std::vector<uint64_t>;

  /// @brief 
  /// @return number of frames in global problem
  auto getGlobalNumberOfFrames() const -> size_t { return num_frames_global_; }

  /// @brief 
  /// @return number of observations in global proglem
  auto getGlobalNumberOfObservations() const -> size_t {
    return num_observations_global_;
  }

  /// @brief 
  /// @return GraphId for current instance of Data 
  auto getGraphId() const -> uint64_t { return graph_id_; }

  /// @brief 
  /// @return vector of Neighbor Ids of the current instance of Data
  auto getNeighbors() const -> std::vector<uint64_t> { return neighbor_ids_; }

  auto writeOutResult(const std::string& ap) -> bool;

 private:
  /// @brief read global properties of Data
  /// @details read properties such as number of frames, map points, obs...
  /// @return bool
  auto readGlobalProperties() -> bool;

  auto getNumberOfLines(const std::string& filename) -> size_t;

  auto readFrames(const std::string& data_folder,
                  const std::string& appendix = "") -> bool;

  auto readMapPoints(const std::string& data_folder,
                     const std::string& appendix = "") -> bool;

  auto readObservations(const std::string& data_folder) -> bool;

  auto readConnectionGraph(const std::string& data_folder) -> bool;

  /// @brief initialize duel weights of the map points
  /// @return bool
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

// Shared and unique pointer
using DataSharedPtr = std::shared_ptr<Data>; // shared pointer let multiple pointer point at the same object!
using DataUniquePtr = std::shared_ptr<Data>;

}  // namespace dba
