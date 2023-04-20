#pragma once

#include <Eigen/Geometry>
#include <memory>
#include <unordered_map>
#include <unordered_set>

#include "distopt/common.hpp"

namespace dba {

class MapPoint {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

 public:
  MapPoint() = delete;
  MapPoint(const uint64_t id, const uint64_t graph_id,
           const Eigen::Vector3d& pos_in_W);
  ~MapPoint();

  auto addObserverId(const uint64_t obs_id) -> void;

  auto setNeighborInvalid(const uint64_t neigh_id) -> void;

  auto isNeighborInvalid(const uint64_t neigh_id) const -> bool;

  auto setDualData(const uint64_t neigh_id, const MapPointDual& dual) -> void;

  auto getDualData(const uint64_t neigh_id, MapPointDual& dual) -> bool;

  auto updateDualVariables() -> bool;

  auto setCommDual(const uint64_t neigh_id, const MapPointDual& dual) -> void;

  auto getCommDual(const uint64_t neigh_id, MapPointDual& dual) -> bool;

  auto setCentralDual(const MapPointDual& dual) -> void;

  auto getCentralDual(MapPointDual& dual) -> bool;

 public:
  // We store the Position as an Eigen Vector in the public domain, as the
  // optimization will directly work on this variable
  Eigen::Vector3d position_;
  bool is_valid_;
  MapPointDual average_state_;
  double sigma_;

  double lambda_;

 private:
  const uint64_t id_;
  const uint64_t graph_id_;
  std::unordered_set<uint64_t> observers_;

  ObservationMap observations_;

  // From here on we store the information required for the distributed
  // optimization (e.g. dual variables etc.)
  Eigen::Quaterniond q_ref_;
  std::unordered_map<uint64_t, MapPointDual> z_hat_;
  std::unordered_map<uint64_t, MapPointDual> z_;
  std::unordered_set<uint64_t> z_invalid_;
  MapPointDual central_dual_;
};

using MapPointSharedPtr = std::shared_ptr<MapPoint>;
using MapPointUniquePtr = std::unique_ptr<MapPoint>;

}  // namespace dba
