#include "linear_model.h"

#include <gtest/gtest.h>

#include <random>
#include <tuple>

using namespace mllib;

std::tuple<MatrixData, VectorTarget> generate_random_data(
    const int n_samples = 100, const int n_features = 2) {
  const double mean = 0.0, std_dev = 1.0;

  std::default_random_engine generator;
  std::normal_distribution<double> distribution(mean, std_dev);

  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> X(n_samples,
                                                          n_features);
  Eigen::Vector<double, Eigen::Dynamic> y(n_samples);

  for (auto sample_it : X.rowwise()) {
    for (auto feature_it = sample_it.begin(); feature_it != sample_it.end();
         ++feature_it) {
      *feature_it = distribution(generator);
    }
  }
  for (auto sample_it = y.begin(); sample_it != y.end(); ++sample_it) {
    *sample_it = distribution(generator);
  }
  return {X, y};
}

TEST(LineaRegression, DefaultConstructor) {
  auto dataset = generate_random_data();
  auto X = std::get<0>(dataset);
  auto y = std::get<1>(dataset);

  linear_model::LinearRegression lr;
  lr.fit(X, y);

  EXPECT_EQ(lr.coef_.size(), X.cols() + 1);
}

TEST(LineaRegression, SolverNormal) {
  const std::string solver = "normal";

  const int n_samples = 10000, n_features = 50;
  auto dataset = generate_random_data(n_samples, n_features);
  auto X = std::get<0>(dataset);
  auto y = std::get<1>(dataset);

  linear_model::LinearRegression lr(true, solver);
  lr.fit(X, y);
  EXPECT_EQ(lr.coef_.size(), X.cols() + 1);

  auto y_pred = lr.predict(X);
  EXPECT_EQ(y_pred.size(), y.size());

  lr = linear_model::LinearRegression(false, solver);
  lr.fit(X, y);
  EXPECT_EQ(lr.coef_.size(), X.cols());

  y_pred = lr.predict(X);
  EXPECT_EQ(y_pred.size(), y.size());
}

TEST(LineaRegression, SolverCholesky) {
  const std::string solver = "cholesky";

  const int n_samples = 10000, n_features = 50;
  auto dataset = generate_random_data(n_samples, n_features);
  auto X = std::get<0>(dataset);
  auto y = std::get<1>(dataset);

  linear_model::LinearRegression lr(true, solver);
  lr.fit(X, y);
  EXPECT_EQ(lr.coef_.size(), X.cols() + 1);

  auto y_pred = lr.predict(X);
  EXPECT_EQ(y_pred.size(), y.size());

  lr = linear_model::LinearRegression(false, solver);
  lr.fit(X, y);
  EXPECT_EQ(lr.coef_.size(), X.cols());

  y_pred = lr.predict(X);
  EXPECT_EQ(y_pred.size(), y.size());
}

TEST(LineaRegression, SolverSVD) {
  const std::string solver = "svd";

  const int n_samples = 10000, n_features = 50;
  auto dataset = generate_random_data(n_samples, n_features);
  auto X = std::get<0>(dataset);
  auto y = std::get<1>(dataset);

  linear_model::LinearRegression lr(true, solver);
  lr.fit(X, y);
  EXPECT_EQ(lr.coef_.size(), X.cols() + 1);

  auto y_pred = lr.predict(X);
  EXPECT_EQ(y_pred.size(), y.size());

  lr = linear_model::LinearRegression(false, solver);
  lr.fit(X, y);
  EXPECT_EQ(lr.coef_.size(), X.cols());

  y_pred = lr.predict(X);
  EXPECT_EQ(y_pred.size(), y.size());
}

TEST(LineaRegression, SolverQR) {
  const std::string solver = "qr";

  const int n_samples = 10000, n_features = 50;
  auto dataset = generate_random_data(n_samples, n_features);
  auto X = std::get<0>(dataset);
  auto y = std::get<1>(dataset);

  linear_model::LinearRegression lr(true, solver);
  lr.fit(X, y);
  EXPECT_EQ(lr.coef_.size(), X.cols() + 1);

  auto y_pred = lr.predict(X);
  EXPECT_EQ(y_pred.size(), y.size());

  lr = linear_model::LinearRegression(false, solver);
  lr.fit(X, y);
  EXPECT_EQ(lr.coef_.size(), X.cols());

  y_pred = lr.predict(X);
  EXPECT_EQ(y_pred.size(), y.size());
}

TEST(LineaRegression, UnknownSolver) {
  const std::string solver = "unkown";

  auto dataset = generate_random_data();
  auto X = std::get<0>(dataset);
  auto y = std::get<1>(dataset);

  linear_model::LinearRegression lr(true, solver);
  EXPECT_THROW(lr.fit(X, y), std::runtime_error);
}
