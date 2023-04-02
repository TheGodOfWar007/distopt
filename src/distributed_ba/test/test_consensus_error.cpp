#include <glog/logging.h>
#include <gtest/gtest.h>
#include <numeric>
#include <random>

#include "distributed_bundle_adjustment/cost_functions/central_consensus.hpp"
#include "distributed_bundle_adjustment/cost_functions/decentral_consensus.hpp"
#include "distributed_bundle_adjustment/utils.hpp"

namespace {

std::random_device r;
std::default_random_engine rand_engine(r());

template <int N>
using EuclideanTypeCentral = dba::cost_functions::CentralEuclideanConsensus<N>;
template <int N>
using EuclideanTypeDecentral =
    dba::cost_functions::DecentralEuclideanConsensus<N>;

TEST(CentralEuclideanConsensus, jacobian) {
  std::uniform_real_distribution<double> uniform_dist_sigma(0.001, 30);
  std::uniform_real_distribution<double> uniform_dist_value(0.2, 0.2);
  std::uniform_real_distribution<double> uniform_dist_init(-5000.0, 5000.0);
  const double delta = 1e-6;
  for (int t = 0; t < 10000; ++t) {
    for (int i = 0; i < 3; ++i) {
      std::unique_ptr<ceres::CostFunction> cost_function;
      const double sigma = uniform_dist_sigma(r);
      double** jacobian_an = new double*[1];
      double** parameter = new double*[1];
      double* residual_nom;
      double* residual_dist;
      double** parameter_dist = new double*[1];
      Eigen::VectorXd value_nom;
      Eigen::MatrixXd J_an;
      int num_vars = 0;
      switch (i) {
        case 0: {
          Eigen::Matrix<double, 1, 1> initial_val;
          initial_val(0, 0) = uniform_dist_sigma(r);
          value_nom.resize(1);
          value_nom[0] = initial_val(0, 0) + uniform_dist_value(r);
          parameter[0] = new double[1]{value_nom[0]};
          jacobian_an[0] = new double[1];
          residual_nom = new double[1];
          residual_dist = new double[1];
          parameter_dist[0] = new double[1];
          cost_function = std::make_unique<EuclideanTypeCentral<1>>(
              sigma, initial_val, initial_val);
          cost_function->Evaluate(parameter, residual_nom, jacobian_an);
          num_vars = 1;
          parameter_dist[0][0] = parameter[0][0] + delta;
          cost_function->Evaluate(parameter_dist, residual_dist, jacobian_an);
          double J_num = (residual_dist[0] - residual_nom[0]) / delta;
          EXPECT_NEAR(J_num, jacobian_an[0][0], 5 * delta);
          break;
        }
        case 1: {
          Eigen::Matrix<double, 3, 1> initial_val;
          J_an.resize(3, 3);
          value_nom.resize(3);
          for (int j = 0; j < 3; ++j) {
            initial_val(j, 0) = uniform_dist_sigma(r);
            value_nom[j] = initial_val(j, 0) + uniform_dist_value(r);
          }
          parameter[0] =
              new double[3]{value_nom[0], value_nom[1], value_nom[2]};
          jacobian_an[0] = new double[3 * 3];
          residual_nom = new double[3];
          residual_dist = new double[3];
          parameter_dist[0] = new double[3];
          cost_function = std::make_unique<EuclideanTypeCentral<3>>(
              sigma, initial_val, initial_val);
          cost_function->Evaluate(parameter, residual_nom, jacobian_an);
          Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> tmp(
              jacobian_an[0]);
          for (int row = 0; row < 3; ++row) {
            for (int col = 0; col < 3; ++col) J_an(row, col) = tmp(row, col);
          }
          num_vars = 3;
          break;
        }
        case 2: {
          Eigen::Matrix<double, 10, 1> initial_val;
          value_nom.resize(10);
          J_an.resize(10, 10);
          for (int j = 0; j < 10; ++j) {
            initial_val(j, 0) = uniform_dist_sigma(r);
            value_nom[j] = initial_val(j, 0) + uniform_dist_value(r);
          }
          parameter[0] = new double[10]{
              value_nom[0], value_nom[1], value_nom[2], value_nom[3],
              value_nom[4], value_nom[5], value_nom[6], value_nom[7],
              value_nom[8], value_nom[9]};
          jacobian_an[0] = new double[10 * 10];
          residual_nom = new double[10];
          residual_dist = new double[10];
          parameter_dist[0] = new double[10];
          cost_function = std::make_unique<EuclideanTypeCentral<10>>(
              sigma, initial_val, initial_val);
          cost_function->Evaluate(parameter, residual_nom, jacobian_an);
          Eigen::Map<Eigen::Matrix<double, 10, 10, Eigen::RowMajor>> tmp(
              jacobian_an[0]);
          for (int row = 0; row < 10; ++row) {
            for (int col = 0; col < 10; ++col) J_an(row, col) = tmp(row, col);
          }
          num_vars = 10;
          break;
        }
      }
      if (num_vars > 1) {
        for (int v = 0; v < num_vars; ++v) {
          for (int x = 0; x < num_vars; ++x)
            parameter_dist[0][x] = parameter[0][x];
          parameter_dist[0][v] += delta;
          cost_function->Evaluate(parameter_dist, residual_dist, nullptr);
          for (int x = 0; x < num_vars; ++x) {
            double J_num = (residual_dist[x] - residual_nom[x]) / delta;
            EXPECT_NEAR(J_num, J_an(x, v), 5 * delta);
          }
        }
      }

      // Free up memory
      delete[] parameter[0];
      delete[] parameter_dist[0];
      delete[] jacobian_an[0];
      delete[] parameter;
      delete[] parameter_dist;
      delete[] jacobian_an;
      delete[] residual_nom;
      delete[] residual_dist;
    }
  }
}

TEST(CentralRotationConsensus, jacobian) {
  std::uniform_real_distribution<double> uniform_dist_sigma(0.001, 30);
  std::uniform_real_distribution<double> uniform_dist_value(0.2, 0.2);
  std::uniform_real_distribution<double> uniform_dist_init(-1.0, 1.0);
  double** jacobian = new double*[1];
  jacobian[0] = new double[4 * 3];
  double** parameter = new double*[1];
  parameter[0] = new double[4];
  double** parameter_dist = new double*[1];
  parameter_dist[0] = new double[4];
  double* residual_nom = new double[3];
  double* residual_dist = new double[3];
  const double delta = 1e-6;
  for (int i = 0; i < 10000; ++i) {
    const double sigma = uniform_dist_sigma(r);
    Eigen::Quaterniond q_ref(uniform_dist_init(r), uniform_dist_init(r),
                             uniform_dist_init(r), uniform_dist_init(r));
    q_ref.normalize();
    Eigen::Vector3d init_dist(uniform_dist_value(r), uniform_dist_value(r),
                              uniform_dist_value(r));
    Eigen::Quaterniond q_val_nom;
    dba::utils::rotmath::Plus(q_ref, init_dist, &q_val_nom);
    parameter[0][0] = q_val_nom.x();
    parameter[0][1] = q_val_nom.y();
    parameter[0][2] = q_val_nom.z();
    parameter[0][3] = q_val_nom.w();

    std::unique_ptr<ceres::CostFunction> cost_function =
        std::make_unique<dba::cost_functions::CentralRotationConsensus>(
            sigma, q_ref, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
    cost_function->Evaluate(parameter, residual_nom, jacobian);
    const Eigen::Map<const Eigen::Matrix<double, 3, 4, Eigen::RowMajor>> J_an(
        jacobian[0]);
    Eigen::Map<Eigen::Quaterniond> q_dist(parameter_dist[0]);
    for (int v = 0; v < 4; ++v) {
      q_dist = q_val_nom;
      parameter_dist[0][v] += delta;
      q_dist.normalize();
      cost_function->Evaluate(parameter_dist, residual_dist, nullptr);
      for (int x = 0; x < 3; ++x) {
        const double J_num = (residual_dist[x] - residual_nom[x]) / delta;
        const double tol = std::max(5 * delta, 1e-5 * std::abs(J_num));
        EXPECT_NEAR(J_num, J_an(x, v), tol);
      }
    }
  }
  delete[] jacobian[0];
  delete[] parameter[0];
  delete[] parameter_dist[0];

  delete[] jacobian;
  delete[] parameter;
  delete[] parameter_dist;
  delete[] residual_nom;
  delete[] residual_dist;
}

TEST(DecentralEuclideanConsensus, jacobian) {
  std::uniform_real_distribution<double> uniform_dist_sigma(0.001, 30);
  std::uniform_real_distribution<double> uniform_dist_value(0.2, 0.2);
  std::uniform_real_distribution<double> uniform_dist_init(-5000.0, 5000.0);
  const double delta = 1e-6;
  for (int t = 0; t < 10000; ++t) {
    for (int i = 0; i < 3; ++i) {
      std::unique_ptr<ceres::CostFunction> cost_function;
      const double lambda = uniform_dist_sigma(r);
      double** jacobian_an = new double*[1];
      double** parameter = new double*[1];
      double* residual_nom;
      double* residual_dist;
      double** parameter_dist = new double*[1];
      Eigen::VectorXd value_nom;
      Eigen::MatrixXd J_an;
      int num_vars = 0;
      switch (i) {
        case 0: {
          Eigen::Matrix<double, 1, 1> initial_val;
          initial_val(0, 0) = uniform_dist_sigma(r);
          value_nom.resize(1);
          value_nom[0] = initial_val(0, 0) + uniform_dist_value(r);
          parameter[0] = new double[1]{value_nom[0]};
          jacobian_an[0] = new double[1];
          residual_nom = new double[1];
          residual_dist = new double[1];
          parameter_dist[0] = new double[1];
          dba::VectorOfVectorN<1> duals;
          for (int k = 0; k < 4; ++k) {
            Eigen::Matrix<double, 1, 1> d;
            d(0, 0) = uniform_dist_sigma(r);
            duals.push_back(d);
          }
          cost_function =
              std::make_unique<EuclideanTypeDecentral<1>>(lambda, duals);
          cost_function->Evaluate(parameter, residual_nom, jacobian_an);
          num_vars = 1;
          parameter_dist[0][0] = parameter[0][0] + delta;
          cost_function->Evaluate(parameter_dist, residual_dist, jacobian_an);
          double J_num = (residual_dist[0] - residual_nom[0]) / delta;
          EXPECT_NEAR(J_num, jacobian_an[0][0], 5 * delta);
          break;
        }
        case 1: {
          Eigen::Matrix<double, 3, 1> initial_val;
          J_an.resize(3, 3);
          value_nom.resize(3);
          for (int j = 0; j < 3; ++j) {
            initial_val(j, 0) = uniform_dist_sigma(r);
            value_nom[j] = initial_val(j, 0) + uniform_dist_value(r);
          }
          parameter[0] =
              new double[3]{value_nom[0], value_nom[1], value_nom[2]};
          jacobian_an[0] = new double[3 * 3];
          residual_nom = new double[3];
          residual_dist = new double[3];
          parameter_dist[0] = new double[3];
          dba::VectorOfVectorN<3> duals;
          for (int k = 0; k < 4; ++k) {
            Eigen::Matrix<double, 3, 1> d;
            d(0, 0) = uniform_dist_sigma(r);
            d(1, 0) = uniform_dist_sigma(r);
            d(2, 0) = uniform_dist_sigma(r);
            duals.push_back(d);
          }
          cost_function =
              std::make_unique<EuclideanTypeDecentral<3>>(lambda, duals);
          cost_function->Evaluate(parameter, residual_nom, jacobian_an);
          Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> tmp(
              jacobian_an[0]);
          for (int row = 0; row < 3; ++row) {
            for (int col = 0; col < 3; ++col) J_an(row, col) = tmp(row, col);
          }
          num_vars = 3;
          break;
        }
        case 2: {
          Eigen::Matrix<double, 10, 1> initial_val;
          value_nom.resize(10);
          J_an.resize(10, 10);
          for (int j = 0; j < 10; ++j) {
            initial_val(j, 0) = uniform_dist_sigma(r);
            value_nom[j] = initial_val(j, 0) + uniform_dist_value(r);
          }
          parameter[0] = new double[10]{
              value_nom[0], value_nom[1], value_nom[2], value_nom[3],
              value_nom[4], value_nom[5], value_nom[6], value_nom[7],
              value_nom[8], value_nom[9]};
          jacobian_an[0] = new double[10 * 10];
          residual_nom = new double[10];
          residual_dist = new double[10];
          parameter_dist[0] = new double[10];
          dba::VectorOfVectorN<10> duals;
          for (int k = 0; k < 4; ++k) {
            Eigen::Matrix<double, 10, 1> d;
            d(0, 0) = uniform_dist_sigma(r);
            d(1, 0) = uniform_dist_sigma(r);
            d(2, 0) = uniform_dist_sigma(r);
            d(3, 0) = uniform_dist_sigma(r);
            d(4, 0) = uniform_dist_sigma(r);
            d(5, 0) = uniform_dist_sigma(r);
            d(6, 0) = uniform_dist_sigma(r);
            d(7, 0) = uniform_dist_sigma(r);
            d(8, 0) = uniform_dist_sigma(r);
            d(9, 0) = uniform_dist_sigma(r);
            duals.push_back(d);
          }
          cost_function =
              std::make_unique<EuclideanTypeDecentral<10>>(lambda, duals);
          cost_function->Evaluate(parameter, residual_nom, jacobian_an);
          Eigen::Map<Eigen::Matrix<double, 10, 10, Eigen::RowMajor>> tmp(
              jacobian_an[0]);
          for (int row = 0; row < 10; ++row) {
            for (int col = 0; col < 10; ++col) J_an(row, col) = tmp(row, col);
          }
          num_vars = 10;
          break;
        }
      }
      if (num_vars > 1) {
        for (int v = 0; v < num_vars; ++v) {
          for (int x = 0; x < num_vars; ++x)
            parameter_dist[0][x] = parameter[0][x];
          parameter_dist[0][v] += delta;
          cost_function->Evaluate(parameter_dist, residual_dist, nullptr);
          for (int x = 0; x < num_vars; ++x) {
            double J_num = (residual_dist[x] - residual_nom[x]) / delta;
            EXPECT_NEAR(J_num, J_an(x, v), 5 * delta);
          }
        }
      }

      // Free up memory
      delete[] parameter[0];
      delete[] parameter_dist[0];
      delete[] jacobian_an[0];
      delete[] parameter;
      delete[] parameter_dist;
      delete[] jacobian_an;
      delete[] residual_nom;
      delete[] residual_dist;
    }
  }
}

TEST(DecentralRotationConsensus, jacobian) {
  std::uniform_real_distribution<double> uniform_dist_sigma(0.001, 30);
  std::uniform_real_distribution<double> uniform_dist_value(0.2, 0.2);
  std::uniform_real_distribution<double> uniform_dist_init(-1.0, 1.0);
  double** jacobian = new double*[1];
  jacobian[0] = new double[4 * 3];
  double** parameter = new double*[1];
  parameter[0] = new double[4];
  double** parameter_dist = new double*[1];
  parameter_dist[0] = new double[4];
  double* residual_nom = new double[3];
  double* residual_dist = new double[3];
  const double delta = 1e-6;
  for (int i = 0; i < 10000; ++i) {
    const double lambda = uniform_dist_sigma(r);
    Eigen::Quaterniond q_ref(uniform_dist_init(r), uniform_dist_init(r),
                             uniform_dist_init(r), uniform_dist_init(r));
    q_ref.normalize();
    Eigen::Vector3d init_dist(uniform_dist_value(r), uniform_dist_value(r),
                              uniform_dist_value(r));
    Eigen::Quaterniond q_val_nom;
    dba::utils::rotmath::Plus(q_ref, init_dist, &q_val_nom);
    parameter[0][0] = q_val_nom.x();
    parameter[0][1] = q_val_nom.y();
    parameter[0][2] = q_val_nom.z();
    parameter[0][3] = q_val_nom.w();
    dba::VectorOfVector3 duals;
    for (int k = 0; k < 4; ++k) {
      Eigen::Vector3d d;
      d[0] = uniform_dist_sigma(r);
      d[1] = uniform_dist_sigma(r);
      d[2] = uniform_dist_sigma(r);
      duals.push_back(d);
    }
    std::unique_ptr<ceres::CostFunction> cost_function =
        std::make_unique<dba::cost_functions::DecentralRotationConsensus>(
            lambda, q_ref, duals);
    cost_function->Evaluate(parameter, residual_nom, jacobian);
    const Eigen::Map<const Eigen::Matrix<double, 3, 4, Eigen::RowMajor>> J_an(
        jacobian[0]);
    Eigen::Map<Eigen::Quaterniond> q_dist(parameter_dist[0]);
    for (int v = 0; v < 4; ++v) {
      q_dist = q_val_nom;
      parameter_dist[0][v] += delta;
      q_dist.normalize();
      cost_function->Evaluate(parameter_dist, residual_dist, nullptr);
      for (int x = 0; x < 3; ++x) {
        const double J_num = (residual_dist[x] - residual_nom[x]) / delta;
        const double tol = std::max(5 * delta, 1e-5 * std::abs(J_num));
        EXPECT_NEAR(J_num, J_an(x, v), tol);
      }
    }
  }
  delete[] jacobian[0];
  delete[] parameter[0];
  delete[] parameter_dist[0];

  delete[] jacobian;
  delete[] parameter;
  delete[] parameter_dist;
  delete[] residual_nom;
  delete[] residual_dist;
}

}  // namespace

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
