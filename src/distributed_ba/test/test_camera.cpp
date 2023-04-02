#include <glog/logging.h>
#include <gtest/gtest.h>
#include <numeric>
#include <random>

#include "distributed_bundle_adjustment/pinhole_camera.hpp"
#include "distributed_bundle_adjustment/rad_distortion.hpp"
#include "distributed_bundle_adjustment/radtan_distortion.hpp"
#include "distributed_bundle_adjustment/simple_pinhole_camera.hpp"

namespace {
std::random_device r;
std::default_random_engine rand_engine(r());

constexpr size_t num_tests = 100;

TEST(SimplePinholeCamera, point_jacobian) {
  for (size_t t = 0; t < num_tests; ++t) {
    std::uniform_real_distribution<double> uniform_dist_point(-15.0, 15);
    std::uniform_real_distribution<double> uniform_dist_param(-0.005, 0.005);
    Eigen::VectorXd dist_params;
    dist_params.resize(2);
    dist_params[0] = uniform_dist_param(r);
    dist_params[1] = uniform_dist_param(r);
    const Eigen::Vector3d point_in_C_nom(
        uniform_dist_point(r), uniform_dist_point(r),
        std::abs(uniform_dist_point(r)) * 10 + 0.01);
    dba::DistortionUniquePtr distortion;
    distortion.reset(new dba::RadDistortion(dist_params));
    Eigen::VectorXd intrinsics;
    intrinsics.resize(1);
    intrinsics[0] = std::abs(uniform_dist_param(r)) * 10000.0;
    dba::SimplePinholeCamera camera(intrinsics, distortion);
    Eigen::Vector2d projection_nom;
    Eigen::Matrix<double, 2, 3> J_wrt_point;
    camera.projectPoint(point_in_C_nom, &projection_nom, &J_wrt_point, nullptr,
                        nullptr);
    const double delta = 1e-5;
    Eigen::Matrix<double, 2, 3> J_wrt_point_num;
    for (size_t i = 0; i < 3; ++i) {
      Eigen::Vector3d point_in_C_dist = point_in_C_nom;
      point_in_C_dist[i] += delta;
      Eigen::Vector2d projection_dist;
      camera.projectPoint(point_in_C_dist, &projection_dist, nullptr, nullptr,
                          nullptr);
      J_wrt_point_num.block<2, 1>(0, i) =
          (projection_dist - projection_nom) / delta;
    }
    for (size_t row = 0; row < 2; ++row) {
      for (size_t col = 0; col < 2; ++col) {
        double allowed_err =
            std::max(delta * 2, 1e-4 * std::abs(J_wrt_point(row, col)));
        EXPECT_NEAR(J_wrt_point(row, col), J_wrt_point_num(row, col),
                    allowed_err);
      }
    }
  }
}

TEST(SimplePinholeCamera, intrinsics_jacobian) {
  for (size_t t = 0; t < num_tests; ++t) {
    std::uniform_real_distribution<double> uniform_dist_point(-15.0, 15);
    std::uniform_real_distribution<double> uniform_dist_param(-0.005, 0.005);
    Eigen::VectorXd dist_params;
    dist_params.resize(2);
    dist_params[0] = uniform_dist_param(r);
    dist_params[1] = uniform_dist_param(r);
    const Eigen::Vector3d point_in_C_nom(
        uniform_dist_point(r), uniform_dist_point(r),
        std::abs(uniform_dist_point(r)) * 5.0 + 0.01);
    dba::DistortionUniquePtr distortion;
    distortion.reset(new dba::RadDistortion(dist_params));
    Eigen::VectorXd intrinsics_nom;
    intrinsics_nom.resize(1);
    intrinsics_nom[0] = std::abs(uniform_dist_param(r)) * 10000.0;
    dba::SimplePinholeCamera camera(intrinsics_nom, distortion);
    Eigen::Vector2d projection_nom;
    Eigen::Matrix<double, 2, Eigen::Dynamic> J_wrt_intrinsics;
    camera.projectPoint(point_in_C_nom, &projection_nom, nullptr,
                        &J_wrt_intrinsics, nullptr);
    const double delta = 1e-5;
    Eigen::Matrix<double, 2, 1> J_wrt_intrinsics_num;
    for (size_t i = 0; i < 1; ++i) {
      Eigen::VectorXd intrinsics_dist = intrinsics_nom;
      intrinsics_dist[i] += delta;
      Eigen::Vector2d projection_dist;
      camera.projectPointUsingExternalParameters(
          intrinsics_dist, dist_params, point_in_C_nom, &projection_dist,
          nullptr, nullptr, nullptr);
      J_wrt_intrinsics_num.block<2, 1>(0, i) =
          (projection_dist - projection_nom) / delta;
    }
    for (size_t row = 0; row < 2; ++row) {
      for (size_t col = 0; col < 1; ++col) {
        double allowed_err = std::max(
            delta * 5, 1e-4 * std::abs(J_wrt_intrinsics_num(row, col)));
        EXPECT_NEAR(J_wrt_intrinsics(row, col), J_wrt_intrinsics_num(row, col),
                    allowed_err);
      }
    }
  }
}

TEST(SimplePinholeCamera, distortion_jacobian) {
  for (size_t t = 0; t < num_tests; ++t) {
    std::uniform_real_distribution<double> uniform_dist_point(-15.0, 15);
    std::uniform_real_distribution<double> uniform_dist_param(-0.005, 0.005);
    Eigen::VectorXd dist_params_nom;
    dist_params_nom.resize(dba::RadDistortion::kNumParameters);
    dist_params_nom[0] = uniform_dist_param(r);
    dist_params_nom[1] = uniform_dist_param(r);
    const Eigen::Vector3d point_in_C_nom(
        uniform_dist_point(r), uniform_dist_point(r),
        std::abs(uniform_dist_point(r)) * 5.0 + 0.01);
    dba::DistortionUniquePtr distortion;
    distortion.reset(new dba::RadDistortion(dist_params_nom));
    Eigen::VectorXd intrinsics;
    intrinsics.resize(1);
    intrinsics[0] = std::abs(uniform_dist_param(r)) * 10000.0;
    dba::SimplePinholeCamera camera(intrinsics, distortion);
    Eigen::Vector2d projection_nom;
    Eigen::Matrix<double, 2, Eigen::Dynamic> J_wrt_dist;
    camera.projectPoint(point_in_C_nom, &projection_nom, nullptr, nullptr,
                        &J_wrt_dist);
    const double delta = 1e-5;
    Eigen::Matrix<double, 2, dba::RadDistortion::kNumParameters> J_wrt_dist_num;
    for (size_t i = 0; i < dba::RadDistortion::kNumParameters; ++i) {
      Eigen::VectorXd dist_params_dist = dist_params_nom;
      dist_params_dist[i] += delta;
      Eigen::Vector2d projection_dist;
      camera.projectPointUsingExternalParameters(
          intrinsics, dist_params_dist, point_in_C_nom, &projection_dist,
          nullptr, nullptr, nullptr);
      J_wrt_dist_num.block<2, 1>(0, i) =
          (projection_dist - projection_nom) / delta;
    }
    for (size_t row = 0; row < 2; ++row) {
      for (size_t col = 0; col < dba::RadDistortion::kNumParameters; ++col) {
        double allowed_err =
            std::max(delta * 5, 1e-4 * std::abs(J_wrt_dist_num(row, col)));
        EXPECT_NEAR(J_wrt_dist(row, col), J_wrt_dist_num(row, col),
                    allowed_err);
      }
    }
  }
}

TEST(PinholeCamera, point_jacobian) {
  for (size_t t = 0; t < num_tests; ++t) {
    std::uniform_real_distribution<double> uniform_dist_point(-15.0, 15);
    std::uniform_real_distribution<double> uniform_dist_param(-0.005, 0.005);
    Eigen::VectorXd intrinsics;
    intrinsics.resize(dba::PinholeCamera::kNumParameters);
    for (size_t i = 0; i < dba::PinholeCamera::kNumParameters; ++i) {
      intrinsics[i] = std::abs(uniform_dist_param(r)) * 10000.0;
    }
    {
      // Radial distortion
      Eigen::VectorXd dist_params;
      dist_params.resize(dba::RadDistortion::kNumParameters);
      for (size_t i = 0; i < dba::RadDistortion::kNumParameters; ++i) {
        dist_params[i] = uniform_dist_param(r);
      }
      const Eigen::Vector3d point_in_C_nom(
          uniform_dist_point(r), uniform_dist_point(r),
          std::abs(uniform_dist_point(r)) * 10 + 0.01);
      dba::DistortionUniquePtr distortion;
      distortion.reset(new dba::RadDistortion(dist_params));
      dba::PinholeCamera camera(intrinsics, distortion);
      Eigen::Vector2d projection_nom;
      Eigen::Matrix<double, 2, 3> J_wrt_point;
      camera.projectPoint(point_in_C_nom, &projection_nom, &J_wrt_point,
                          nullptr, nullptr);
      const double delta = 1e-6;
      Eigen::Matrix<double, 2, 3> J_wrt_point_num;
      for (size_t i = 0; i < 3; ++i) {
        Eigen::Vector3d point_in_C_dist = point_in_C_nom;
        point_in_C_dist[i] += delta;
        Eigen::Vector2d projection_dist;
        camera.projectPoint(point_in_C_dist, &projection_dist, nullptr, nullptr,
                            nullptr);
        J_wrt_point_num.block<2, 1>(0, i) =
            (projection_dist - projection_nom) / delta;
      }
      for (size_t row = 0; row < 2; ++row) {
        for (size_t col = 0; col < 3; ++col) {
          double allowed_err =
              std::max(delta * 2, 5e-4 * std::abs(J_wrt_point(row, col)));
          EXPECT_NEAR(J_wrt_point(row, col), J_wrt_point_num(row, col),
                      allowed_err);
        }
      }
    }
    {
      // Radial-Tangential distortion
      Eigen::VectorXd dist_params;
      dist_params.resize(dba::RadTanDistortion::kNumParameters);
      for (size_t i = 0; i < dba::RadTanDistortion::kNumParameters; ++i) {
        dist_params[i] = uniform_dist_param(r);
      }
      const Eigen::Vector3d point_in_C_nom(
          uniform_dist_point(r), uniform_dist_point(r),
          std::abs(uniform_dist_point(r)) * 10 + 0.01);
      dba::DistortionUniquePtr distortion;
      distortion.reset(new dba::RadTanDistortion(dist_params));
      dba::PinholeCamera camera(intrinsics, distortion);
      Eigen::Vector2d projection_nom;
      Eigen::Matrix<double, 2, 3> J_wrt_point;
      camera.projectPoint(point_in_C_nom, &projection_nom, &J_wrt_point,
                          nullptr, nullptr);
      const double delta = 1e-5;
      Eigen::Matrix<double, 2, 3> J_wrt_point_num;
      for (size_t i = 0; i < 3; ++i) {
        Eigen::Vector3d point_in_C_dist = point_in_C_nom;
        point_in_C_dist[i] += delta;
        Eigen::Vector2d projection_dist;
        camera.projectPoint(point_in_C_dist, &projection_dist, nullptr, nullptr,
                            nullptr);
        J_wrt_point_num.block<2, 1>(0, i) =
            (projection_dist - projection_nom) / delta;
      }
      for (size_t row = 0; row < 2; ++row) {
        for (size_t col = 0; col < 3; ++col) {
          double allowed_err =
              std::max(delta * 2, 5e-4 * std::abs(J_wrt_point(row, col)));
          EXPECT_NEAR(J_wrt_point(row, col), J_wrt_point_num(row, col),
                      allowed_err);
        }
      }
    }
  }
}

TEST(PinholeCamera, intrinsics_jacobian) {
  for (size_t t = 0; t < num_tests; ++t) {
    std::uniform_real_distribution<double> uniform_dist_point(-15.0, 15);
    std::uniform_real_distribution<double> uniform_dist_param(-0.001, 0.001);
    Eigen::VectorXd intrinsics_nom;
    intrinsics_nom.resize(dba::PinholeCamera::kNumParameters);
    intrinsics_nom[0] = 800.0;
    intrinsics_nom[1] = 800.0;
    intrinsics_nom[2] = 450.0;
    intrinsics_nom[3] = 500.0;
    for (size_t i = 0; i < dba::PinholeCamera::kNumParameters; ++i) {
      intrinsics_nom[i] += std::abs(uniform_dist_param(r)) * 3000.0;
    }
    const Eigen::Vector3d point_in_C_nom(
        uniform_dist_point(r), uniform_dist_point(r),
        std::abs(uniform_dist_point(r)) * 5.0 + 0.01);
    {
      Eigen::VectorXd dist_params;
      dist_params.resize(dba::RadDistortion::kNumParameters);
      for (size_t i = 0; i < dba::RadDistortion::kNumParameters; ++i) {
        dist_params[i] = uniform_dist_param(r);
      }
      dba::DistortionUniquePtr distortion;
      distortion.reset(new dba::RadDistortion(dist_params));
      dba::PinholeCamera camera(intrinsics_nom, distortion);
      Eigen::Vector2d projection_nom;
      Eigen::Matrix<double, 2, Eigen::Dynamic> J_wrt_intrinsics;
      auto res = camera.projectPoint(point_in_C_nom, &projection_nom, nullptr,
                                     &J_wrt_intrinsics, nullptr);
      if (res != dba::Camera::ProjectionResult::SUCCESSFUL) continue;
      const double delta = 1e-6;
      Eigen::Matrix<double, 2, dba::PinholeCamera::kNumParameters>
          J_wrt_intrinsics_num;
      for (size_t i = 0; i < dba::PinholeCamera::kNumParameters; ++i) {
        Eigen::VectorXd intrinsics_dist = intrinsics_nom;
        intrinsics_dist[i] += delta;
        Eigen::Vector2d projection_dist;
        auto res_dist = camera.projectPointUsingExternalParameters(
            intrinsics_dist, dist_params, point_in_C_nom, &projection_dist,
            nullptr, nullptr, nullptr);
        J_wrt_intrinsics_num.block<2, 1>(0, i) =
            (projection_dist - projection_nom) / delta;
        if (res_dist != dba::Camera::ProjectionResult::SUCCESSFUL)
          std::cout << "Fails projection in disturbance step...";
      }
      for (size_t row = 0; row < 2; ++row) {
        for (size_t col = 0; col < dba::PinholeCamera::kNumParameters; ++col) {
          double allowed_err = std::max(
              delta * 5, 5e-4 * std::abs(J_wrt_intrinsics_num(row, col)));
          EXPECT_NEAR(J_wrt_intrinsics(row, col),
                      J_wrt_intrinsics_num(row, col), allowed_err)
              << "row: " << row << ", col: " << col;
        }
      }
    }
    {
      Eigen::VectorXd dist_params;
      dist_params.resize(dba::RadTanDistortion::kNumParameters);
      for (size_t i = 0; i < dba::RadTanDistortion::kNumParameters; ++i) {
        dist_params[i] = uniform_dist_param(r);
      }
      dba::DistortionUniquePtr distortion;
      distortion.reset(new dba::RadTanDistortion(dist_params));
      dba::PinholeCamera camera(intrinsics_nom, distortion);
      Eigen::Vector2d projection_nom;
      Eigen::Matrix<double, 2, Eigen::Dynamic> J_wrt_intrinsics;
      auto res = camera.projectPoint(point_in_C_nom, &projection_nom, nullptr,
                                     &J_wrt_intrinsics, nullptr);
      if (res != dba::Camera::ProjectionResult::SUCCESSFUL) continue;
      const double delta = 1e-6;
      Eigen::Matrix<double, 2, dba::PinholeCamera::kNumParameters>
          J_wrt_intrinsics_num;
      for (size_t i = 0; i < dba::PinholeCamera::kNumParameters; ++i) {
        Eigen::VectorXd intrinsics_dist = intrinsics_nom;
        intrinsics_dist[i] += delta;
        Eigen::Vector2d projection_dist;
        camera.projectPointUsingExternalParameters(
            intrinsics_dist, dist_params, point_in_C_nom, &projection_dist,
            nullptr, nullptr, nullptr);
        J_wrt_intrinsics_num.block<2, 1>(0, i) =
            (projection_dist - projection_nom) / delta;
      }
      for (size_t row = 0; row < 2; ++row) {
        for (size_t col = 0; col < dba::PinholeCamera::kNumParameters; ++col) {
          double allowed_err = std::max(
              delta * 5, 5e-4 * std::abs(J_wrt_intrinsics_num(row, col)));
          EXPECT_NEAR(J_wrt_intrinsics(row, col),
                      J_wrt_intrinsics_num(row, col), allowed_err)
              << "row: " << row << ", col: " << col;
        }
      }
    }
  }
}

TEST(PinholeCamera, distortion_jacobian) {
  for (size_t t = 0; t < num_tests; ++t) {
    std::uniform_real_distribution<double> uniform_dist_point(-15.0, 15);
    std::uniform_real_distribution<double> uniform_dist_param(-0.005, 0.005);
    Eigen::VectorXd intrinsics;
    intrinsics.resize(dba::PinholeCamera::kNumParameters);
    intrinsics[0] = 800.0;
    intrinsics[1] = 800.0;
    intrinsics[2] = 450.0;
    intrinsics[3] = 500.0;
    for (size_t i = 0; i < dba::PinholeCamera::kNumParameters; ++i) {
      intrinsics[i] += std::abs(uniform_dist_param(r)) * 3000.0;
    }
    const Eigen::Vector3d point_in_C_nom(
        uniform_dist_point(r), uniform_dist_point(r),
        std::abs(uniform_dist_point(r)) * 5.0 + 0.01);
    {
      // Radial Distortion
      Eigen::VectorXd dist_params_nom;
      dist_params_nom.resize(dba::RadDistortion::kNumParameters);
      for (size_t i = 0; i < dba::RadDistortion::kNumParameters; ++i) {
        dist_params_nom[i] = uniform_dist_param(r);
      }
      dba::DistortionUniquePtr distortion;
      distortion.reset(new dba::RadDistortion(dist_params_nom));
      dba::PinholeCamera camera(intrinsics, distortion);
      Eigen::Vector2d projection_nom;
      Eigen::Matrix<double, 2, Eigen::Dynamic> J_wrt_dist;
      auto res = camera.projectPoint(point_in_C_nom, &projection_nom, nullptr,
                                     nullptr, &J_wrt_dist);
      if (res != dba::Camera::ProjectionResult::SUCCESSFUL) continue;
      const double delta = 1e-6;
      Eigen::Matrix<double, 2, dba::RadDistortion::kNumParameters>
          J_wrt_dist_num;
      for (size_t i = 0; i < dba::RadDistortion::kNumParameters; ++i) {
        Eigen::VectorXd dist_params_dist = dist_params_nom;
        dist_params_dist[i] += delta;
        Eigen::Vector2d projection_dist;
        auto res2 = camera.projectPointUsingExternalParameters(
            intrinsics, dist_params_dist, point_in_C_nom, &projection_dist,
            nullptr, nullptr, nullptr);
        if (res2 != dba::Camera::ProjectionResult::SUCCESSFUL) continue;
        J_wrt_dist_num.block<2, 1>(0, i) =
            (projection_dist - projection_nom) / delta;
      }
      for (size_t row = 0; row < 2; ++row) {
        for (size_t col = 0; col < dba::RadDistortion::kNumParameters; ++col) {
          double allowed_err =
              std::max(delta * 5, 5e-4 * std::abs(J_wrt_dist_num(row, col)));
          EXPECT_NEAR(J_wrt_dist(row, col), J_wrt_dist_num(row, col),
                      allowed_err);
        }
      }
    }
    {
      // RadialTangential Distortion
      Eigen::VectorXd dist_params_nom;
      dist_params_nom.resize(dba::RadTanDistortion::kNumParameters);
      for (size_t i = 0; i < dba::RadTanDistortion::kNumParameters; ++i) {
        dist_params_nom[i] = uniform_dist_param(r);
      }

      dba::DistortionUniquePtr distortion;
      distortion.reset(new dba::RadTanDistortion(dist_params_nom));
      dba::PinholeCamera camera(intrinsics, distortion);
      Eigen::Vector2d projection_nom;
      Eigen::Matrix<double, 2, Eigen::Dynamic> J_wrt_dist;
      auto res = camera.projectPoint(point_in_C_nom, &projection_nom, nullptr,
                                     nullptr, &J_wrt_dist);
      if (res != dba::Camera::ProjectionResult::SUCCESSFUL) continue;
      const double delta = 1e-6;
      Eigen::Matrix<double, 2, dba::RadTanDistortion::kNumParameters>
          J_wrt_dist_num;
      for (size_t i = 0; i < dba::RadTanDistortion::kNumParameters; ++i) {
        Eigen::VectorXd dist_params_dist = dist_params_nom;
        dist_params_dist[i] += delta;
        Eigen::Vector2d projection_dist;
        auto res2 = camera.projectPointUsingExternalParameters(
            intrinsics, dist_params_dist, point_in_C_nom, &projection_dist,
            nullptr, nullptr, nullptr);
        if (res2 != dba::Camera::ProjectionResult::SUCCESSFUL) continue;
        J_wrt_dist_num.block<2, 1>(0, i) =
            (projection_dist - projection_nom) / delta;
      }
      for (size_t row = 0; row < 2; ++row) {
        for (size_t col = 0; col < dba::RadTanDistortion::kNumParameters;
             ++col) {
          double allowed_err =
              std::max(delta * 5, 5e-4 * std::abs(J_wrt_dist_num(row, col)));
          EXPECT_NEAR(J_wrt_dist(row, col), J_wrt_dist_num(row, col),
                      allowed_err)
              << "row: " << row << ", col: " << col;
        }
      }
    }
  }
}

}  // namespace

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
