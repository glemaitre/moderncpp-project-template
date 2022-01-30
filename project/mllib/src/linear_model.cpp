#include "linear_model.h"

#include <iostream>

using namespace mllib;
using namespace linear_model;

LinearRegression::LinearRegression(const bool fit_intercept,
                                   const std::string solver)
    : fit_intercept(fit_intercept), solver(solver) {}

LinearRegression::~LinearRegression() {}

void LinearRegression::fit(const MatrixData &X, const VectorTarget &y) {
  if (solver == "normal")
    this->coef_ = (X.transpose() * X).inverse() * X.transpose() * y;
}