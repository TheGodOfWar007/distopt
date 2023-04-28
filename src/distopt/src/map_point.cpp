#include <gflags/gflags.h>
#include <glog/logging.h>

#include "distopt/map_point.hpp"
#include "distopt/rad_distortion.hpp"
#include "distopt/simple_pinhole_camera.hpp"
#include "distopt/utils.hpp"

DEFINE_double(alpha_map_points, 1.0e5,
              "The consensus weight parameter for the map points");

DECLARE_double(asynch_nu);

namespace dba {

MapPoint::MapPoint(const uint64_t id, const uint64_t graph_id,
                   const Eigen::Vector3d& pos_in_W)
    : is_valid_(true), id_(id), graph_id_(graph_id), position_(pos_in_W) {}

MapPoint::~MapPoint() {}

auto MapPoint::addObserverId(const uint64_t obs_id) -> void {
  observers_.insert(obs_id);
}

auto MapPoint::setNeighborInvalid(const uint64_t neigh_id) -> void {
  z_invalid_.insert(neigh_id);
}

auto MapPoint::isNeighborInvalid(const uint64_t neigh_id) const -> bool {
  return z_invalid_.count(neigh_id);
}

auto MapPoint::setDualData(const uint64_t neigh_id, const MapPointDual& dual)
    -> void {
  if (!z_.count(neigh_id)) z_[neigh_id] = MapPointDual(id_);
  auto& z = z_[neigh_id];
  for (size_t i = 0; i < z.getSize(); ++i) z[i] = dual[i];
}

auto MapPoint::getDualData(const uint64_t neigh_id, MapPointDual& dual)
    -> bool {
  if (!z_.count(neigh_id)) return false;
  const auto z = z_[neigh_id];
  for (size_t i = 0; i < z.getSize(); ++i) dual[i] = z[i];
  return true;
}

auto MapPoint::updateDualVariables() -> bool {
  if (z_hat_.empty()) return false;
  MapPointDual x_lambda;
  Eigen::Map<Eigen::Vector3d> dual_pos(x_lambda.getPosition());
  dual_pos = lambda_ * position_;
  for (const auto& [id, z_hat_i] : z_hat_) {
    if (!z_.count(id)) return false;
    auto& z_i = z_[id];
    for (size_t i = 0; i < x_lambda.getSize(); ++i)
      z_i[i] -= FLAGS_asynch_nu * ((z_i[i] + z_hat_i[i]) * 0.5 + x_lambda[i]);
  }
  return true;
}

auto MapPoint::setCommDual(const uint64_t neigh_id, const MapPointDual& dual)
    -> void {
  if (!z_hat_.count(neigh_id)) z_hat_[neigh_id] = MapPointDual(id_);
  auto& z_hat = z_hat_[neigh_id];
  for (size_t i = 0; i < z_hat.getSize(); ++i) z_hat[i] = dual[i];
}

auto MapPoint::getCommDual(const uint64_t neigh_id, MapPointDual& dual)
    -> bool {
  if (!z_hat_.count(neigh_id)) return false;
  const auto& z_hat = z_hat_[neigh_id];
  for (size_t i = 0; i < z_hat.getSize(); ++i) dual[i] = z_hat[i];
  return true;
}

auto MapPoint::setCentralDual(const MapPointDual& dual) -> void {
  for (size_t i = 0; i < dual.getSize(); ++i) central_dual_[i] = dual[i];
}

auto MapPoint::getCentralDual(MapPointDual& dual) -> bool {
  //  if (dual[0] > 1e12) return false;
  for (size_t i = 0; i < dual.getSize(); ++i) dual[i] = central_dual_[i];
  return true;
}

}  // namespace dba
