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
}