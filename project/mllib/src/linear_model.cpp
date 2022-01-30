#include "linear_model.h"

#include <iostream>

using namespace mllib;
using namespace linear_model;

LinearRegression::LinearRegression(bool fit_intercept)
    : fit_intercept(fit_intercept) {}

LinearRegression::~LinearRegression() {}

void LinearRegression::fit(const MatrixData &X, const VectorTarget &y) {
  std::cout << "Start fitting the linear regression..." << std::endl;
}