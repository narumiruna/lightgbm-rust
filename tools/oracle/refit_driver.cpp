// Observe actual native RefitTree derivatives without replacing native kernels.
// See THIRD_PARTY_NOTICES.md.
#include "driver_common.hpp"
#include <LightGBM/c_api.h>
#include <LightGBM/objective_function.h>
#include <LightGBM/tree.h>
#include "src/boosting/gbdt.h"
#include "src/boosting/dart.hpp"
#include "src/boosting/rf.hpp"
#include <memory>
#include <sstream>
using namespace LightGBM;

template<class T> std::vector<T> values(const J& j) {
  std::vector<T> out;
  for (const auto& v : j.array_items()) out.push_back(static_cast<T>(v.number_value()));
  return out;
}
std::string config_string(const J& params) {
  std::string result;
  for (const auto& p : params.object_items()) {
    result += p.first + "=";
    if (p.second.is_array()) {
      bool first = true;
      for (const auto& v : p.second.array_items()) { if (!first) result += ","; result += v.is_string() ? v.string_value() : v.dump(); first = false; }
    } else result += p.second.is_string() ? p.second.string_value() : p.second.dump();
    result += " ";
  }
  return result;
}
void checked(int status) { if (status) throw std::runtime_error(LGBM_GetLastError()); }

template<class Base> class RefitProbe : public Base {
 public:
  A steps;
  bool collecting = false;
  std::vector<double> scores() const {
    const double* scores = this->train_score_updater_->score();
    return std::vector<double>(scores, scores + this->num_data_ * this->num_tree_per_iteration_);
  }
 protected:
  void Boosting() override {
    // Do not call virtual GetTrainingScore() here: DART uses it to drop trees.
    auto before = collecting ? scores() : std::vector<double>();
    Base::Boosting();
    if (!collecting) return;  // RF::Init also invokes Boosting().
    size_t size = before.size();
    std::vector<double> grad(this->gradients_pointer_, this->gradients_pointer_ + size);
    std::vector<double> hess(this->hessians_pointer_, this->hessians_pointer_ + size);
    steps.emplace_back(O{{"iteration", J(static_cast<int>(steps.size()))},
                         {"training_scores_before", finite_values(before)},
                         {"gradients", finite_values(grad)}, {"hessians", finite_values(hess)}});
  }
};

template<class Base> J refit(const J& c, const Config& config, const Dataset* data, const ObjectiveFunction* objective) {
  std::ifstream model_file(c["initial_model_path"].string_value()); std::stringstream model; model << model_file.rdbuf();
  std::string model_text = model.str();
  GBDT source; source.LoadModelFromString(model_text.c_str(), model_text.size());
  RefitProbe<Base> target; target.Init(&config, data, objective, {}); target.MergeFrom(&source);
  std::vector<int> leaf_pred;
  for (const auto& row : c["leaf_pred"].array_items()) for (const auto& v : row.array_items()) leaf_pred.push_back(v.int_value());
  size_t trees = c["leaf_pred"][0].array_items().size();
  target.collecting = true;
  target.RefitTree(leaf_pred.data(), data->num_data(), trees);
  target.SaveModelToFile(0, -1, 0, c["output_model_path"].string_value().c_str());
  return J(O{{"name", c["name"]}, {"steps", J(target.steps)},
             {"training_scores_after", finite_values(target.scores())}});
}

int main(int argc, char** argv) {
  if (argc != 3) return 2;
  static_assert(sizeof(score_t) == 4 && sizeof(label_t) == 4);
  Log::ResetLogLevel(LogLevel::Fatal);
  std::ifstream input(argv[1]); std::stringstream buffer; buffer << input.rdbuf(); std::string error;
  J request = J::parse(buffer.str(), &error);
  if (!error.empty()) throw std::runtime_error(error);
  A cases;
  for (const auto& c : request["cases"].array_items()) {
    std::string params = config_string(c["params"]);
    Config config; config.Set(Config::Str2Map(params.c_str()));
    // Text-imported trees do not restore leaf_parent_; smoothing can read invalid memory.
    if (config.path_smooth > kEpsilon) throw std::runtime_error("unsafe imported-tree path_smooth refit is not an oracle case");
    auto labels = values<label_t>(c["labels"]), weights = values<label_t>(c["weights"]);
    auto groups = values<int32_t>(c["groups"]);
    int n = static_cast<int>(labels.size()), features = static_cast<int>(c["rows"][0].array_items().size());
    std::vector<double> rows;
    for (const auto& row : c["rows"].array_items()) for (const auto& v : row.array_items()) rows.push_back(v.number_value());
    DatasetHandle handle;
    checked(LGBM_DatasetCreateFromMat(rows.data(), C_API_DTYPE_FLOAT64, n, features, 1, params.c_str(), nullptr, &handle));
    checked(LGBM_DatasetSetField(handle, "label", labels.data(), n, C_API_DTYPE_FLOAT32));
    if (!weights.empty()) checked(LGBM_DatasetSetField(handle, "weight", weights.data(), n, C_API_DTYPE_FLOAT32));
    if (!groups.empty()) checked(LGBM_DatasetSetField(handle, "group", groups.data(), static_cast<int>(groups.size()), C_API_DTYPE_INT32));
    const Dataset* data = reinterpret_cast<Dataset*>(handle);
    std::unique_ptr<ObjectiveFunction> objective(ObjectiveFunction::CreateObjectiveFunction(config.objective, config));
    objective->Init(data->metadata(), n);
    if (config.boosting == "rf") cases.push_back(refit<RF>(c, config, data, objective.get()));
    else if (config.boosting == "dart") cases.push_back(refit<DART>(c, config, data, objective.get()));
    else cases.push_back(refit<GBDT>(c, config, data, objective.get()));
    checked(LGBM_DatasetFree(handle));
  }
  save(argv[2], "refit_internal", cases);
}
