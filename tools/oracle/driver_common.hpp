// Reference-only JSON output utilities. See THIRD_PARTY_NOTICES.md.
#pragma once
#include <LightGBM/utils/json11.h>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

using J = json11_internal_lightgbm::Json;
using O = J::object;
using A = J::array;
const char* commit = "5d84f3f222b41409a6983d2ac90513ad9702fa9e";

void save(const std::string& dir, const std::string& name, const A& cases) {
  std::ofstream out(dir + "/" + name + ".json");
  out << J(O{{"schema_version", J(1)}, {"upstream_commit", J(commit)}, {"cases", J(cases)}}).dump() << '\n';
  if (!out) throw std::runtime_error("cannot write fixture");
}
J finite_values(const std::vector<double>& values, bool bounds = false) {
  A result;
  for (double v : values) {
    if (std::isnan(v)) result.emplace_back(bounds ? J("NaN") : J(nullptr));
    else if (std::isinf(v)) result.emplace_back(v > 0 ? "Infinity" : "-Infinity");
    else result.emplace_back(v);
  }
  return J(result);
}
