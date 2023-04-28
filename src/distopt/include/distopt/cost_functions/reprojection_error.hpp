#pragma once

#include <ceres/sized_cost_function.h>
#include <glog/logging.h>
#include <Eigen/Dense>
#include <memory>

#include "distopt/camera.hpp"
#include "distopt/distortion.hpp"
#include "distopt/simple_pinhole_camera.hpp"
#include "distopt/utils.hpp"

namespace dba {

namespace cost_functions {

template <class CameraType, class DistortionType>
class ReprojectionError
    : public ceres::SizedCostFunction<2, 3, 4, 3, CameraType::kNumParameters,
                                      DistortionType::kNumParameters> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

 public:
  ReprojectionError() = delete;
  ReprojectionError(const Eigen::Vector2d observation, const double sigma,
                    const CameraType* camera_ptr)
      : observation_(observation),
        sigma_inv_(1.0 / sigma),
        camera_ptr_(camera_ptr) {
    CHECK(camera_ptr != nullptr);
  };
  ~ReprojectionError(){};

  virtual auto Evaluate(double const* const* parameters, double* residuals,
                        double** jacobians) const -> bool;

 private:
  const Eigen::Vector2d observation_;
  const double sigma_inv_;
  const CameraType* camera_ptr_;
};

// The following class was used to investigate a strange behaviour when enabling
// the gradient checking option. But it turned out that the gradient checker
// uses the "RIDDLING" method, which for some reason seems to fail at certain
// points. Since when switching to CENTRAL the results matched the analytical
// ones.
class ReprojectionErrorAuto {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

 public:
  ReprojectionErrorAuto() = delete;
  ReprojectionErrorAuto(const Eigen::Vector2d& observation, const double sigma,
                        const dba::SimplePinholeCamera* camera_ptr)
      : observation_(observation),
        sigma_inv_(1.0 / sigma),
        camera_ptr_(camera_ptr) {
    CHECK(camera_ptr != nullptr);
  };
  ~ReprojectionErrorAuto(){};

  bool operator()(const double* const point_in_W_, const double* const q_W_C_,
                  const double* const p_W_C_, const double* const intrinsics_,
                  const double* const dist_params_, double* residuals) const {
    Eigen::Map<const Eigen::Vector3d> p_in_W(point_in_W_);
    Eigen::Map<const Eigen::Quaterniond> q_W_C(q_W_C_);
    Eigen::Map<const Eigen::Vector3d> p_W_C(p_W_C_);
    Eigen::Map<const Eigen::Matrix<double, 1, 1>> intrinsics_map(intrinsics_);
    Eigen::Map<const Eigen::Matrix<double, 2, 1>> distortion_map(dist_params_);
    Eigen::VectorXd intrinsics = intrinsics_map;
    Eigen::VectorXd distortion = distortion_map;
    const Eigen::Matrix3d R_W_C = q_W_C.toRotationMatrix();
    const Eigen::Matrix3d R_C_W = R_W_C.transpose();
    const Eigen::Vector3d p_in_C = R_C_W * (p_in_W - p_W_C);
    Eigen::Vector2d proj;
    auto projection = camera_ptr_->projectPointUsingExternalParameters(
        intrinsics, distortion, p_in_C, &proj, nullptr, nullptr, nullptr);
    Eigen::Map<Eigen::Vector2d> residual(residuals);
    if (projection == Camera::ProjectionResult::SUCCESSFUL) {
      residual = (proj - observation_) * this->sigma_inv_;
    } else {
      residual.setZero();
    }
    return true;
  }

 private:
  const Eigen::Vector2d observation_;
  const double sigma_inv_;
  const dba::SimplePinholeCamera* camera_ptr_;
};

}  // namespace cost_functions

}  // namespace dba

#include "reprojection_error_impl.hpp"
