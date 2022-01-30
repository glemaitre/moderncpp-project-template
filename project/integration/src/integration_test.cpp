#include "linear_model.h"

#include <iostream>

using namespace mllib;

int main() {
  linear_model::LinearRegression lr(false);
  std::cout << lr.fit_intercept << std::endl;
  std::cout << "Hello, World!\n" << std::endl;
  return 0;
}