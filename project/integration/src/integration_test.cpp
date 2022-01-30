#include <Eigen/Dense>
#include <iostream>
#include <random>

#include "linear_model.h"

using namespace mllib;

int main() {
  const int n_samples = 100, n_features = 2;
  const double mean = 0.0, std_dev = 1.0;

  std::default_random_engine generator;
  std::normal_distribution<double> distribution(mean, std_dev);

  Eigen::Matrix<double, n_samples, n_features> X;
  Eigen::Vector<double, n_samples> y;

  for (auto sample_it : X.rowwise()) {
    for (auto feature_it = sample_it.begin(); feature_it != sample_it.end();
         ++feature_it) {
      *feature_it = distribution(generator);
    }
  }
  for (auto sample_it = y.begin(); sample_it != y.end(); ++sample_it) {
    *sample_it = distribution(generator);
  }

  linear_model::LinearRegression lr(false);
  std::cout << lr.coef_ << std::endl;
  lr.fit(X, y);
  std::cout << lr.coef_ << std::endl;
  return 0;
}