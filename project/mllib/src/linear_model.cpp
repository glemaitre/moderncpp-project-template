#include "linear_model.h"

#include <iostream>

using namespace mllib;
using namespace linear_model;

LinearRegression::LinearRegression(const bool fit_intercept,
                                   const std::string solver)
    : fit_intercept(fit_intercept), solver(solver) {}

LinearRegression::~LinearRegression() {}

void LinearRegression::fit(const MatrixData &X, const VectorTarget &y) {
  auto X_ = X;
  if (fit_intercept) {
    // add a dummy column
    X_.conservativeResize(Eigen::NoChange, X_.cols() + 1);
    X_.col(X_.cols() - 1).setOnes();
  }

  if (solver == "normal")
    this->coef_ = (X_.transpose() * X_).inverse() * X_.transpose() * y;
  else if (solver == "cholesky")
    this->coef_ = (X_.transpose() * X_).ldlt().solve(X_.transpose() * y);
  else if (solver == "svd")
    this->coef_ = X_.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(y);
  else if (solver == "qr")
    this->coef_ = X_.colPivHouseholderQr().solve(y);
  else
    throw std::runtime_error("unknown solver");
}

VectorTarget LinearRegression::predict(const MatrixData &X) {
  auto X_ = X;
  if (fit_intercept) {
    // add a dummy column
    X_.conservativeResize(Eigen::NoChange, X_.cols() + 1);
    X_.col(X_.cols() - 1).setOnes();
  }

  return X_ * this->coef_;
}