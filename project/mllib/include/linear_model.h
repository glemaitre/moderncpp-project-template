#include <optional>

#include "mllib_export.h"
#include "common.h"

namespace mllib {
namespace linear_model {
class MLLIB_EXPORT LinearRegression {
 public:
  bool fit_intercept;

  VectorParameter coef_;

 public:
  LinearRegression(const bool fit_intercept = true);
  ~LinearRegression();

  void fit(const MatrixData &X, const VectorTarget &y);
};
}  // namespace linear_model
}  // namespace mllib