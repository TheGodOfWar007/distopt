#include <glog/logging.h>
#include <gtest/gtest.h>
#include <numeric>
#include <random>

#include "distributed_bundle_adjustment/equidistant_distortion.hpp"
#include "distributed_bundle_adjustment/rad_distortion.hpp"
#include "distributed_bundle_adjustment/radtan_distortion.hpp"

namespace {
std::random_device r;
std::default_random_engine rand_engine(r());
constexpr size_t num_samples = 10000;

TEST(RadDistortion, point_jacobian) {
  for (size_t t = 0; t < num_samples; ++t) {
    std::uniform_real_distribution<double> uniform_dist_point(-0.8, 0.8);
    std::uniform_real_distribution<double> uniform_dist_param(-0.8, 0.8);
    Eigen::VectorXd dist_params;
    dist_params.resize(dba::RadDistortion::kNumParameters);
    for (size_t i = 0; i < dba::RadDistortion::kNumParameters; ++i) {
      dist_params[i] = uniform_dist_param(r);
    }
    const Eigen::Vector2d point_in_nom(uniform_dist_point(r),
                                       uniform_dist_point(r));
    dba::RadDistortion rad_dist(dist_params);
    Eigen::Vector2d point_out_nom;
    Eigen::Matrix2d J_wrt_point;
    rad_dist.distortPoint(point_in_nom, &point_out_nom, &J_wrt_point, nullptr);
    Eigen::Vector2d point_out_dist;
    Eigen::Matrix2d J_wrt_point_num = Eigen::Matrix2d::Zero();
    const double delta = 1e-6;
    for (int i = 0; i < 2; ++i) {
      Eigen::Vector2d dx = Eigen::Vector2d::Zero();
      dx[i] = delta;
      const Eigen::Vector2d point_in_dist = point_in_nom + dx;
      Eigen::Vector2d point_out_dist;
      rad_dist.distortPoint(point_in_dist, &point_out_dist, nullptr, nullptr);
      J_wrt_point_num.block<2, 1>(0, i) =
          (point_out_dist - point_out_nom) / delta;
    }
    for (size_t row = 0; row < 2; ++row) {
      for (size_t col = 0; col < 2; ++col) {
        EXPECT_NEAR(J_wrt_point(row, col), J_wrt_point_num(row, col),
                    10.0 * delta);
      }
    }
  }
}

TEST(RadDistortion, dist_param_jacobian) {
  for (size_t t = 0; t < num_samples; ++t) {
    std::uniform_real_distribution<double> uniform_dist_point(-0.8, 0.8);
    std::uniform_real_distribution<double> uniform_dist_param(-0.8, 0.8);
    Eigen::VectorXd dist_params_nom;
    dist_params_nom.resize(dba::RadDistortion::kNumParameters);
    for (size_t i = 0; i < dba::RadDistortion::kNumParameters; ++i) {
      dist_params_nom[i] = uniform_dist_param(r);
    }
    const Eigen::Vector2d point_in(uniform_dist_point(r),
                                   uniform_dist_point(r));
    dba::RadDistortion rad_dist(dist_params_nom);
    Eigen::Vector2d point_out_nom;
    Eigen::Matrix<double, 2, Eigen::Dynamic> J_wrt_dist;
    rad_dist.distortPoint(point_in, &point_out_nom, nullptr, &J_wrt_dist);
    Eigen::Vector2d point_out_dist;
    Eigen::Matrix<double, 2, Eigen::Dynamic> J_wrt_dist_num;
    J_wrt_dist_num.setZero(2, dba::RadDistortion::kNumParameters);
    const double delta = 1e-6;
    for (size_t i = 0; i < 2; ++i) {
      Eigen::VectorXd dist_param_dist = dist_params_nom;
      dist_param_dist[i] += delta;
      Eigen::Vector2d point_out_dist;
      rad_dist.distortPointUsingExternalParameters(
          dist_param_dist, point_in, &point_out_dist, nullptr, nullptr);
      J_wrt_dist_num.block<2, 1>(0, i) =
          (point_out_dist - point_out_nom) / delta;
    }
    for (size_t row = 0; row < 2; ++row) {
      for (size_t col = 0; col < dba::RadDistortion::kNumParameters; ++col) {
        EXPECT_NEAR(J_wrt_dist(row, col), J_wrt_dist_num(row, col),
                    10.0 * delta);
      }
    }
  }
}

TEST(RadTanDistortion, point_jacobian) {
  for (size_t t = 0; t < num_samples; ++t) {
    std::uniform_real_distribution<double> uniform_dist_point(-0.8, 0.8);
    std::uniform_real_distribution<double> uniform_dist_param(-0.2, 0.2);
    Eigen::VectorXd dist_params;
    dist_params.resize(dba::RadTanDistortion::kNumParameters);
    for (size_t i = 0; i < dba::RadTanDistortion::kNumParameters; ++i) {
      dist_params[i] = uniform_dist_param(r);
    }
    const Eigen::Vector2d point_in_nom(uniform_dist_point(r),
                                       uniform_dist_point(r));
    dba::RadTanDistortion rad_dist(dist_params);
    Eigen::Vector2d point_out_nom;
    Eigen::Matrix2d J_wrt_point;
    rad_dist.distortPoint(point_in_nom, &point_out_nom, &J_wrt_point, nullptr);
    Eigen::Vector2d point_out_dist;
    Eigen::Matrix2d J_wrt_point_num = Eigen::Matrix2d::Zero();
    const double delta = 1e-6;
    for (size_t i = 0; i < 2; ++i) {
      Eigen::Vector2d dx = Eigen::Vector2d::Zero();
      dx[i] = delta;
      const Eigen::Vector2d point_in_dist = point_in_nom + dx;
      Eigen::Vector2d point_out_dist;
      rad_dist.distortPoint(point_in_dist, &point_out_dist, nullptr, nullptr);
      J_wrt_point_num.block<2, 1>(0, i) =
          (point_out_dist - point_out_nom) / delta;
    }
    for (size_t row = 0; row < 2; ++row) {
      for (size_t col = 0; col < 2; ++col) {
        EXPECT_NEAR(J_wrt_point(row, col), J_wrt_point_num(row, col),
                    10.0 * delta);
      }
    }
  }
}

TEST(RadTanDistortion, dist_param_jacobian) {
  for (size_t t = 0; t < num_samples; ++t) {
    std::uniform_real_distribution<double> uniform_dist_point(-0.8, 0.8);
    std::uniform_real_distribution<double> uniform_dist_param(-0.2, 0.2);
    Eigen::VectorXd dist_params_nom;
    dist_params_nom.resize(dba::RadTanDistortion::kNumParameters);
    for (size_t i = 0; i < dba::RadTanDistortion::kNumParameters; ++i) {
      dist_params_nom[i] = uniform_dist_param(r);
    }
    const Eigen::Vector2d point_in(uniform_dist_point(r),
                                   uniform_dist_point(r));
    dba::RadTanDistortion rad_dist(dist_params_nom);
    Eigen::Vector2d point_out_nom;
    Eigen::Matrix<double, 2, Eigen::Dynamic> J_wrt_dist;
    rad_dist.distortPoint(point_in, &point_out_nom, nullptr, &J_wrt_dist);
    Eigen::Vector2d point_out_dist;
    Eigen::Matrix<double, 2, Eigen::Dynamic> J_wrt_dist_num;
    J_wrt_dist_num.setZero(2, dba::RadTanDistortion::kNumParameters);
    const double delta = 1e-6;
    for (size_t i = 0; i < dba::RadTanDistortion::kNumParameters; ++i) {
      Eigen::VectorXd dist_param_dist = dist_params_nom;
      dist_param_dist[i] += delta;
      Eigen::Vector2d point_out_dist;
      rad_dist.distortPointUsingExternalParameters(
          dist_param_dist, point_in, &point_out_dist, nullptr, nullptr);
      J_wrt_dist_num.block<2, 1>(0, i) =
          (point_out_dist - point_out_nom) / delta;
    }
    for (size_t row = 0; row < 2; ++row) {
      for (size_t col = 0; col < dba::RadTanDistortion::kNumParameters; ++col) {
        EXPECT_NEAR(J_wrt_dist(row, col), J_wrt_dist_num(row, col),
                    10.0 * delta);
      }
    }
  }
}

TEST(EquidistantDistortion, point_jacobian) {
  for (size_t t = 0; t < num_samples; ++t) {
    std::uniform_real_distribution<double> uniform_dist_point(-0.8, 0.8);
    std::uniform_real_distribution<double> uniform_dist_param(-0.2, 0.2);
    Eigen::VectorXd dist_params;
    dist_params.resize(dba::EquidistantDistortion::kNumParameters);
    for (size_t i = 0; i < dba::EquidistantDistortion::kNumParameters; ++i) {
      dist_params[i] = uniform_dist_param(r);
    }
    const Eigen::Vector2d point_in_nom(uniform_dist_point(r),
                                       uniform_dist_point(r));
    dba::EquidistantDistortion rad_dist(dist_params);
    Eigen::Vector2d point_out_nom;
    Eigen::Matrix2d J_wrt_point;
    rad_dist.distortPoint(point_in_nom, &point_out_nom, &J_wrt_point, nullptr);
    Eigen::Vector2d point_out_dist;
    Eigen::Matrix2d J_wrt_point_num = Eigen::Matrix2d::Zero();
    const double delta = 1e-6;
    for (size_t i = 0; i < 2; ++i) {
      Eigen::Vector2d dx = Eigen::Vector2d::Zero();
      dx[i] = delta;
      const Eigen::Vector2d point_in_dist = point_in_nom + dx;
      Eigen::Vector2d point_out_dist;
      rad_dist.distortPoint(point_in_dist, &point_out_dist, nullptr, nullptr);
      J_wrt_point_num.block<2, 1>(0, i) =
          (point_out_dist - point_out_nom) / delta;
    }
    for (size_t row = 0; row < 2; ++row) {
      for (size_t col = 0; col < 2; ++col) {
        EXPECT_NEAR(J_wrt_point(row, col), J_wrt_point_num(row, col),
                    10.0 * delta);
      }
    }
  }
}

TEST(EquidistantDistortion, dist_param_jacobian) {
  for (size_t t = 0; t < num_samples; ++t) {
    std::uniform_real_distribution<double> uniform_dist_point(-0.8, 0.8);
    std::uniform_real_distribution<double> uniform_dist_param(-0.2, 0.2);
    Eigen::VectorXd dist_params_nom;
    dist_params_nom.resize(dba::EquidistantDistortion::kNumParameters);
    for (size_t i = 0; i < dba::EquidistantDistortion::kNumParameters; ++i) {
      dist_params_nom[i] = uniform_dist_param(r);
    }
    const Eigen::Vector2d point_in(uniform_dist_point(r),
                                   uniform_dist_point(r));
    dba::EquidistantDistortion rad_dist(dist_params_nom);
    Eigen::Vector2d point_out_nom;
    Eigen::Matrix<double, 2, Eigen::Dynamic> J_wrt_dist;
    rad_dist.distortPoint(point_in, &point_out_nom, nullptr, &J_wrt_dist);
    Eigen::Vector2d point_out_dist;
    Eigen::Matrix<double, 2, Eigen::Dynamic> J_wrt_dist_num;
    J_wrt_dist_num.setZero(2, dba::EquidistantDistortion::kNumParameters);
    const double delta = 1e-6;
    for (size_t i = 0; i < dba::EquidistantDistortion::kNumParameters; ++i) {
      Eigen::VectorXd dist_param_dist = dist_params_nom;
      dist_param_dist[i] += delta;
      Eigen::Vector2d point_out_dist;
      rad_dist.distortPointUsingExternalParameters(
          dist_param_dist, point_in, &point_out_dist, nullptr, nullptr);
      J_wrt_dist_num.block<2, 1>(0, i) =
          (point_out_dist - point_out_nom) / delta;
    }
    for (size_t row = 0; row < 2; ++row) {
      for (size_t col = 0; col < dba::EquidistantDistortion::kNumParameters;
           ++col) {
        EXPECT_NEAR(J_wrt_dist(row, col), J_wrt_dist_num(row, col),
                    10.0 * delta);
      }
    }
  }
}

}  // namespace

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
