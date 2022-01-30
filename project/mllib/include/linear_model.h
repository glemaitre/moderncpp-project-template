#include <optional>

#include <Eigen/Dense>

#include "mllib_export.h"
#include "common.h"

namespace mllib {
namespace linear_model {
class MLLIB_EXPORT LinearRegression {
 public:
  bool fit_intercept;

 public:
  LinearRegression(const bool fit_intercept = true);
  ~LinearRegression();

  return fit()
};
}  // namespace linear_model
}  // namespace mllib