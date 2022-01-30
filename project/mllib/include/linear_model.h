#include <optional>
#include <string>

#include "common.h"
#include "mllib_export.h"

namespace mllib {
namespace linear_model {
class MLLIB_EXPORT LinearRegression {
 public:
  bool fit_intercept;
  std::string solver;

  VectorParameter coef_;

 public:
  LinearRegression(const bool fit_intercept = true,
                   const std::string solver = "normal");
  ~LinearRegression();

  void fit(const MatrixData &X, const VectorTarget &y);
};
}  // namespace linear_model
}  // namespace mllib