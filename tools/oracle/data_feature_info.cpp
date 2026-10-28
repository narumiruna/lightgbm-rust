// Development-only fixture generator; pinned LightGBM MIT sources supply BinMapper.
#include <LightGBM/bin.h>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>
using namespace LightGBM;
int main() {
  struct Case { const char* name; std::vector<double> rows; bool categorical; };
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double inf = std::numeric_limits<double>::infinity();
  const double adjacent = std::nextafter(std::nextafter(1., inf), inf);
  const std::vector<Case> cases = {
    {"integers", {-3., 0., 1., 5.}, false},
    {"decimal", {-0.1, 0.7}, false},
    {"scientific_small", {1e-5, 2e-5}, false},
    {"fixed_small", {1e-4, 2e-4}, false},
    {"fixed_large", {1e16, 2e16}, false},
    {"scientific_large", {1e17, 2e17}, false},
    {"all_nan", {nan, nan}, false},
    {"zeros", {-0., 0., 0.}, false},
    {"adjacent", {1., std::nextafter(1., inf), adjacent}, false},
    {"negative_and_implicit_zero", {-3., -1., 0.}, false},
    {"infinite_extrema", {-inf, inf, nan, 0.}, false},
    {"categories", {0., 1., 1., 2., -1., nan}, true},
    {"categories_all_missing", {nan, nan}, true},
    {"categories_all_negative", {-1., -2.}, true}
  };
  for (const auto& c : cases) {
    std::vector<double> sample;
    for (double v : c.rows) if (std::fabs(v) > kZeroThreshold || std::isnan(v)) sample.push_back(v);
    BinMapper mapper;
    mapper.FindBin(sample.data(), static_cast<int>(sample.size()), c.rows.size(), 63, 1, 1, false,
        c.categorical ? BinType::CategoricalBin : BinType::NumericalBin, true, false, {});
    std::cout << c.name << ' ' << std::quoted(mapper.bin_info_string()) << '\n';
  }
}
