namespace dba {

namespace cost_functions {

template <class CameraType, class DistortionType>
auto ReprojectionError<CameraType, DistortionType>::Evaluate(
    double const* const* parameters, double* residuals,
    double** jacobians) const -> bool {
  Eigen::Map<const Eigen::Vector3d> p_in_W(parameters[0]);
  Eigen::Map<const Eigen::Quaterniond> q_W_C(parameters[1]);
  Eigen::Map<const Eigen::Vector3d> p_W_C(parameters[2]);
  Eigen::Map<const Eigen::Matrix<double, CameraType::kNumParameters, 1>>
      intrinsics_map(parameters[3]);
  Eigen::Map<const Eigen::Matrix<double, DistortionType::kNumParameters, 1>>
      distortion_map(parameters[4]);
  using PointJacobian = Eigen::Matrix<double, 2, 3, Eigen::RowMajor>;
  using RotationJacobian = Eigen::Matrix<double, 2, 4, Eigen::RowMajor>;
  using IntrinsicsJacobian = Eigen::Matrix<
      double, 2, CameraType::kNumParameters>;  // here we have to leave out
                                               // RowMajor due to the fact the
                                               // camera parameters are only 1d
  using DistortionJacobian =
      Eigen::Matrix<double, 2, DistortionType::kNumParameters, Eigen::RowMajor>;
  using JacWrtCamParams = Eigen::Matrix<double, 2, Eigen::Dynamic>;
  JacWrtCamParams J_keyp_wrt_intrinsics(2, CameraType::kNumParameters);
  JacWrtCamParams J_keyp_wrt_distortion(2, DistortionType::kNumParameters);
  Eigen::Matrix<double, 2, 3> J_keyp_wrt_p_in_C;

  JacWrtCamParams* J_keyp_wrt_intrinsics_ptr = nullptr;
  JacWrtCamParams* J_keyp_wrt_distortion_ptr = nullptr;
  Eigen::Matrix<double, 2, 3>* J_keyp_wrt_p_in_C_ptr = nullptr;
  if (jacobians && jacobians[0]) J_keyp_wrt_p_in_C_ptr = &J_keyp_wrt_p_in_C;
  if (jacobians && jacobians[3])
    J_keyp_wrt_intrinsics_ptr = &J_keyp_wrt_intrinsics;
  if (jacobians && jacobians[4])
    J_keyp_wrt_distortion_ptr = &J_keyp_wrt_distortion;

  Eigen::VectorXd intrinsics = intrinsics_map;
  Eigen::VectorXd distortion = distortion_map;
  const Eigen::Matrix3d R_C_W = (q_W_C.inverse()).toRotationMatrix();
  const Eigen::Vector3d p_in_C = R_C_W * (p_in_W - p_W_C);
  Eigen::Vector2d proj;
  auto projection = camera_ptr_->projectPointUsingExternalParameters(
      intrinsics, distortion, p_in_C, &proj, J_keyp_wrt_p_in_C_ptr,
      J_keyp_wrt_intrinsics_ptr, J_keyp_wrt_distortion_ptr);

  if (jacobians) {
    utils::QuaternionLocalParameterization quat_parameterization;
    if (jacobians[0]) {
      Eigen::Map<PointJacobian> J_res_wrt_p_in_W(jacobians[0]);
      if (projection == Camera::ProjectionResult::SUCCESSFUL) {
        J_res_wrt_p_in_W = J_keyp_wrt_p_in_C * R_C_W * this->sigma_inv_;
      } else {
        J_res_wrt_p_in_W.setZero();
      }
    }
    if (jacobians[1]) {
      Eigen::Map<RotationJacobian> J_res_wrt_q_W_C(jacobians[1]);
      if (projection == Camera::ProjectionResult::SUCCESSFUL) {
        Eigen::Matrix<double, 4, 3, Eigen::RowMajor> J_quat_local_param;
        quat_parameterization.ComputeJacobian(q_W_C.coeffs().data(),
                                              J_quat_local_param.data());
        Eigen::Matrix3d J_p_in_C_wrt_q_W_C = utils::rotmath::skew(p_in_C);
        J_res_wrt_q_W_C = J_keyp_wrt_p_in_C * J_p_in_C_wrt_q_W_C * 4.0 *
                          J_quat_local_param.transpose() * this->sigma_inv_;
      } else {
        J_res_wrt_q_W_C.setZero();
      }
    }
    if (jacobians[2]) {
      Eigen::Map<PointJacobian> J_res_wrt_p_W_C(jacobians[2]);
      if (projection == Camera::ProjectionResult::SUCCESSFUL) {
        J_res_wrt_p_W_C = -J_keyp_wrt_p_in_C * R_C_W * this->sigma_inv_;
      } else {
        J_res_wrt_p_W_C.setZero();
      }
    }
    if (jacobians[3]) {
      Eigen::Map<IntrinsicsJacobian> J_res_wrt_intrinsics(jacobians[3]);
      if (projection == Camera::ProjectionResult::SUCCESSFUL) {
        J_res_wrt_intrinsics = J_keyp_wrt_intrinsics * this->sigma_inv_;
      } else {
        J_res_wrt_intrinsics.setZero();
      }
    }
    if (jacobians[4]) {
      Eigen::Map<DistortionJacobian> J_res_wrt_distortion(jacobians[4]);
      if (projection == Camera::ProjectionResult::SUCCESSFUL) {
        J_res_wrt_distortion = J_keyp_wrt_distortion * this->sigma_inv_;
      } else {
        J_res_wrt_distortion.setZero();
      }
    }
  }

  Eigen::Map<Eigen::Vector2d> residual(residuals);
  if (projection == Camera::ProjectionResult::SUCCESSFUL) {
    residual = (proj - observation_) * this->sigma_inv_;
  } else {
    residual.setZero();
  }

  return true;
}

}  // namespace cost_functions

}  // namespace dba
