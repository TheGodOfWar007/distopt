#pragma once

#include <Eigen/Geometry>
#include <memory>
#include <unordered_map>
#include <unordered_set>

#include "distributed_bundle_adjustment/camera.hpp"
#include "distributed_bundle_adjustment/common.hpp"
#include "distributed_bundle_adjustment/distortion.hpp"

namespace dba {

class Frame {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  using FrameDual = PoseDual<kNumIntrinsicParams, kNumDistortionParams>;

 public:
  Frame() = delete;
  Frame(const uint64_t id, const uint64_t graph_id,
        const Eigen::Vector3d& p_W_C, const Eigen::Quaterniond& q_W_C,
        const Eigen::VectorXd& intrinsics, const Eigen::VectorXd& dist_coeffs,
        const Camera::Type& cam_type, const Distortion::Type& dist_type);
  ~Frame();

  auto setNeighborInvalid(const uint64_t neigh_id) -> void;

  auto isNeighborInvalid(const uint64_t neigh_id) const -> bool;

  auto setDualData(const uint64_t neigh_id, const FrameDual& dual) -> void;

  auto getDualData(const uint64_t neigh_id, FrameDual& dual) -> bool;

  auto updateDualVariables() -> bool;

  auto setCommDual(const uint64_t neigh_id, const FrameDual& dual) -> void;

  auto getCommDual(const uint64_t neigh_id, FrameDual& dual) -> bool;

  auto setCentralDual(const FrameDual& dual) -> void;

  auto getCentralDual(FrameDual& dual) -> bool;

  auto getReferenceRotation() const -> Eigen::Quaterniond { return q_ref_; }

  auto addObservation(const Observation& obs) -> bool;

  auto getAllObservations() -> ObservationVector;

  auto getCameraType() const -> Camera::Type { return cam_type_; }

  auto getDistortionType() const -> Distortion::Type { return dist_type_; }

  auto getCamera() const -> CameraSharedPtr;

 public:
  // The current pose. As we work directly on them in the optimization, we store
  // them accordingly in the public domain
  Eigen::Vector3d p_W_C_;
  Eigen::Quaterniond q_W_C_;
  // The camera parameters are also defined public as we optimize directly on
  // them
  Eigen::VectorXd intrinsics_;
  Eigen::VectorXd dist_coeffs_;
  bool is_valid_;
  FrameDual average_state_;

  // Store the sigmas (for adaptation and easy access)
  double sigma_trans_;
  double sigma_rot_;
  double sigma_intr_;
  double sigma_dist_;

  // Store the lamdbas (for adaptation and easy access)
  double lambda_trans_;
  double lambda_rot_;
  double lambda_intr_;
  double lambda_dist_;

 private:
  ObservationMap observations_;

  // From here on we store the information required for the distributed
  // optimization (e.g. dual variables etc.)
  Eigen::Quaterniond q_ref_;
  std::unordered_map<uint64_t, FrameDual> z_hat_;
  std::unordered_map<uint64_t, FrameDual> z_;
  std::unordered_set<uint64_t> z_invalid_;
  const uint64_t id_;
  const uint64_t graph_id_;
  FrameDual central_dual_;

  CameraSharedPtr camera_ptr_;
  const Camera::Type cam_type_;
  const Distortion::Type dist_type_;
};

using FrameSharedPtr = std::shared_ptr<Frame>;
using FrameUniquePtr = std::unique_ptr<Frame>;

}  // namespace dba
