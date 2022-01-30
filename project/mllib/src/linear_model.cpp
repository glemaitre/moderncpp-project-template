#include "linear_model.h"

using namespace mllib;
using namespace linear_model;

LinearRegression::LinearRegression(bool fit_intercept)
    : fit_intercept(fit_intercept) {}

LinearRegression::~LinearRegression() {}
