#include <glog/logging.h>
#include <gtest/gtest.h>
#include <numeric>
#include <random>

#include "distributed_bundle_adjustment/cost_functions/reprojection_error.hpp"
#include "distributed_bundle_adjustment/rad_distortion.hpp"
#include "distributed_bundle_adjustment/simple_pinhole_camera.hpp"

namespace {
std::random_device r;
std::default_random_engine rand_engine(r());

TEST(ReprojectionError, pose_jacobian) {
  for (int i = 0; i < 1000; ++i) {
    std::uniform_real_distribution<double> uniform_dist_position(-100.0, 100.0);
    std::uniform_real_distribution<double> uniform_dist_point(-15.0, 15);
    std::uniform_real_distribution<double> uniform_dist_rot(-1.0, 1.0);
    std::uniform_real_distribution<double> uniform_dist_param(-0.005, 0.005);
    Eigen::VectorXd dist_params;
    dist_params.resize(2);
    dist_params[0] = uniform_dist_param(r);
    dist_params[1] = uniform_dist_param(r);
    Eigen::VectorXd intrinsics;
    intrinsics.resize(1);
    intrinsics[0] = std::abs(uniform_dist_param(r)) * 10000.0;
    dba::DistortionUniquePtr distortion;
    distortion.reset(new dba::RadDistortion(dist_params));
    dba::SimplePinholeCamera camera(intrinsics, distortion);
    Eigen::Vector3d p_W_C(uniform_dist_position(r), uniform_dist_position(r),
                          uniform_dist_position(r));
    Eigen::Quaterniond q_W_C(uniform_dist_position(r), uniform_dist_position(r),
                             uniform_dist_position(r),
                             uniform_dist_position(r));
    q_W_C.normalize();
    const Eigen::Vector3d p_in_C(uniform_dist_point(r) * 0.5,
                                 uniform_dist_point(r) * 0.5,
                                 std::abs(uniform_dist_point(r)) * 10 + 0.2);
    Eigen::Vector3d p_in_W = q_W_C * p_in_C + p_W_C;
    Eigen::Vector2d observation;
    auto res =
        camera.projectPoint(p_in_C, &observation, nullptr, nullptr, nullptr);
    if (res != dba::Camera::ProjectionResult::SUCCESSFUL) continue;
    observation[0] += uniform_dist_param(r) * 10.0;
    observation[1] += uniform_dist_param(r) * 10.0;
    double** parameters_nom = new double*[5];
    parameters_nom[0] = new double[3]{p_in_W.x(), p_in_W.y(), p_in_W.z()};
    parameters_nom[1] =
        new double[4]{q_W_C.x(), q_W_C.y(), q_W_C.z(), q_W_C.w()};
    parameters_nom[2] = new double[3]{p_W_C.x(), p_W_C.y(), p_W_C.z()};
    parameters_nom[3] =
        new double[dba::SimplePinholeCamera::kNumParameters]{intrinsics[0]};
    parameters_nom[4] = new double[dba::RadDistortion::kNumParameters]{
        dist_params[0], dist_params[1]};
    double** parameters_dist = new double*[5];
    parameters_dist[0] = new double[3]{p_in_C.x(), p_in_C.y(), p_in_C.z()};
    parameters_dist[1] =
        new double[4]{q_W_C.x(), q_W_C.y(), q_W_C.z(), q_W_C.w()};
    parameters_dist[2] = new double[3]{p_W_C.x(), p_W_C.y(), p_W_C.z()};
    parameters_dist[3] =
        new double[dba::SimplePinholeCamera::kNumParameters]{intrinsics[0]};
    parameters_dist[4] = new double[dba::RadDistortion::kNumParameters]{
        dist_params[0], dist_params[1]};

    double** jacobians_analytical = new double*[5];
    jacobians_analytical[0] = new double[2 * 3];
    jacobians_analytical[1] = new double[2 * 4];
    jacobians_analytical[2] = new double[2 * 3];
    jacobians_analytical[3] =
        new double[2 * dba::SimplePinholeCamera::kNumParameters];
    jacobians_analytical[4] =
        new double[2 * dba::RadDistortion::kNumParameters];
    Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>>
        J_keypoint_wrt_p_in_W(jacobians_analytical[0]);
    Eigen::Map<Eigen::Matrix<double, 2, 4, Eigen::RowMajor>>
        J_keypoint_wrt_q_W_C(jacobians_analytical[1]);
    Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>>
        J_keypoint_wrt_p_W_C(jacobians_analytical[2]);
    Eigen::Map<
        Eigen::Matrix<double, 2,
                      dba::SimplePinholeCamera::
                          kNumParameters>>  // Since kNumParameters is one, we
                                            // must leave out RowMajor
        J_keypoint_wrt_intrinsics(jacobians_analytical[3]);
    Eigen::Map<Eigen::Matrix<double, 2, dba::RadDistortion::kNumParameters,
                             Eigen::RowMajor>>
        J_keypoint_wrt_distortion(jacobians_analytical[4]);

    dba::cost_functions::ReprojectionError<dba::SimplePinholeCamera,
                                           dba::RadDistortion>
        error_term(observation, 1.5, &camera);
    double* residuals_nom = new double[2];
    error_term.Evaluate(parameters_nom, residuals_nom, jacobians_analytical);
    Eigen::Map<Eigen::Vector2d> resid_nom(residuals_nom);

    // Map the residuals
    double* residuals_num = new double[2];
    const double delta = 1e-7;
    Eigen::Map<Eigen::Vector2d> resid_num(residuals_num);
    Eigen::Map<Eigen::Vector3d> p_in_W_num(parameters_dist[0]);
    p_in_W_num = p_in_W;
    Eigen::Map<Eigen::Quaterniond> q_W_C_num(parameters_dist[1]);
    q_W_C_num = q_W_C;
    Eigen::Map<Eigen::Vector3d> p_W_C_num(parameters_dist[2]);
    p_W_C_num = p_W_C;
    Eigen::Map<
        Eigen::Matrix<double, dba::SimplePinholeCamera::kNumParameters, 1>>
        intrinsics_num(parameters_dist[3]);
    intrinsics_num = intrinsics;
    Eigen::Map<Eigen::Matrix<double, dba::RadDistortion::kNumParameters, 1>>
        dist_param_num(parameters_dist[4]);
    dist_param_num = dist_params;
    Eigen::Matrix<double, 2, 3> J_keypoint_wrt_p_in_W_num;
    for (int i = 0; i < 3; ++i) {
      p_in_W_num = p_in_W;
      p_in_W_num[i] += delta;
      error_term.Evaluate(parameters_dist, residuals_num, nullptr);
      J_keypoint_wrt_p_in_W_num.block<2, 1>(0, i) =
          (resid_num - resid_nom) / delta;
    }
    p_in_W_num = p_in_W;

    Eigen::Matrix<double, 2, 4> J_keypoint_wrt_q_W_C_num;
    for (int i = 0; i < 4; ++i) {
      q_W_C_num = q_W_C;
      parameters_dist[1][i] += delta;
      q_W_C_num.normalize();
      error_term.Evaluate(parameters_dist, residuals_num, nullptr);
      J_keypoint_wrt_q_W_C_num.block<2, 1>(0, i) =
          (resid_num - resid_nom) / delta;
    }
    q_W_C_num = q_W_C;

    Eigen::Matrix<double, 2, 3> J_keypoint_wrt_p_W_C_num;
    for (int i = 0; i < 3; ++i) {
      p_W_C_num = p_W_C;
      p_W_C_num[i] += delta;
      error_term.Evaluate(parameters_dist, residuals_num, nullptr);
      J_keypoint_wrt_p_W_C_num.block<2, 1>(0, i) =
          (resid_num - resid_nom) / delta;
    }
    p_W_C_num = p_W_C;

    Eigen::Matrix<double, 2, dba::SimplePinholeCamera::kNumParameters>
        J_keypoint_wrt_intrinsics_num;
    for (int i = 0; i < dba::SimplePinholeCamera::kNumParameters; ++i) {
      intrinsics_num = intrinsics;
      intrinsics_num[i] += delta;
      error_term.Evaluate(parameters_dist, residuals_num, nullptr);
      J_keypoint_wrt_intrinsics_num.block<2, 1>(0, i) =
          (resid_num - resid_nom) / delta;
    }
    intrinsics_num = intrinsics;

    Eigen::Matrix<double, 2, dba::RadDistortion::kNumParameters>
        J_keypoint_wrt_distortion_num;
    for (int i = 0; i < dba::RadDistortion::kNumParameters; ++i) {
      dist_param_num = dist_params;
      dist_param_num[i] += delta;
      error_term.Evaluate(parameters_dist, residuals_num, nullptr);
      J_keypoint_wrt_distortion_num.block<2, 1>(0, i) =
          (resid_num - resid_nom) / delta;
    }
    dist_param_num = dist_params;

    // Check the result
    int range = std::max({4, dba::SimplePinholeCamera::kNumParameters,
                          dba::RadDistortion::kNumParameters});
    for (int row = 0; row < 2; ++row) {
      for (int col = 0; col < range; ++col) {
        if (col < 3) {
          EXPECT_NEAR(
              J_keypoint_wrt_p_in_W(row, col),
              J_keypoint_wrt_p_in_W_num(row, col),
              std::max(std::abs(J_keypoint_wrt_p_in_W_num(row, col)) * 1e-3,
                       1e-5));
          EXPECT_NEAR(
              J_keypoint_wrt_p_W_C(row, col),
              J_keypoint_wrt_p_W_C_num(row, col),
              std::max(std::abs(J_keypoint_wrt_p_W_C_num(row, col)) * 1e-3,
                       1e-5));
        }
        if (col < 4) {
          EXPECT_NEAR(
              J_keypoint_wrt_q_W_C(row, col),
              J_keypoint_wrt_q_W_C_num(row, col),
              std::max(std::abs(J_keypoint_wrt_q_W_C_num(row, col)) * 1e-3,
                       1e-4));
        }
        if (col < dba::SimplePinholeCamera::kNumParameters) {
          EXPECT_NEAR(
              J_keypoint_wrt_intrinsics(row, col),
              J_keypoint_wrt_intrinsics_num(row, col),
              std::max(std::abs(J_keypoint_wrt_intrinsics_num(row, col)) * 1e-3,
                       5e-5));
        }
        if (col < dba::RadDistortion::kNumParameters) {
          EXPECT_NEAR(
              J_keypoint_wrt_distortion(row, col),
              J_keypoint_wrt_distortion_num(row, col),
              std::max(std::abs(J_keypoint_wrt_distortion_num(row, col)) * 1e-3,
                       5e-5));
        }
      }
    }

    // Free up memory
    for (int i = 0; i < 5; ++i) {
      delete[] parameters_nom[i];
      delete[] parameters_dist[i];
      delete[] jacobians_analytical[i];
    }
    delete[] parameters_nom;
    delete[] parameters_dist;
    delete[] residuals_nom;
    delete[] jacobians_analytical;
    delete[] residuals_num;
  }
}

}  // namespace

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
