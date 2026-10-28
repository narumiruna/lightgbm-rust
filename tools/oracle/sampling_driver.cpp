// Native-only internal sampling and quantization evidence. See THIRD_PARTY_NOTICES.md.
#include "driver_common.hpp"
#include <LightGBM/c_api.h>
#include <LightGBM/sample_strategy.h>
#include <LightGBM/tree.h>
#include <LightGBM/metric.h>
#include "src/treelearner/col_sampler.hpp"
#include "src/treelearner/gradient_discretizer.hpp"
#include "src/boosting/gbdt.h"
#include "src/boosting/score_updater.hpp"
// Development probe only: expose DART's recorded state, without replacing its code.
#define private public
#include "src/boosting/dart.hpp"
#undef private
#include <sstream>
#include <memory>
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
    auto labels = values<label_t>(c["labels"]), weights = values<label_t>(c["weights"]);
    auto groups = values<int32_t>(c["groups"]), positions = values<int32_t>(c["positions"]);
    const int n = static_cast<int>(labels.size()), features = static_cast<int>(c["rows"][0].array_items().size());
    std::vector<double> rows;
    for (const auto& row : c["rows"].array_items()) for (const auto& value : row.array_items()) rows.push_back(value.is_null() ? std::nan("") : value.number_value());
    DatasetHandle handle;
    checked(LGBM_DatasetCreateFromMat(rows.data(), C_API_DTYPE_FLOAT64, n, features, 1, params.c_str(), nullptr, &handle));
    checked(LGBM_DatasetSetField(handle, "label", labels.data(), n, C_API_DTYPE_FLOAT32));
    if (!weights.empty()) checked(LGBM_DatasetSetField(handle, "weight", weights.data(), n, C_API_DTYPE_FLOAT32));
    if (!groups.empty()) checked(LGBM_DatasetSetField(handle, "group", groups.data(), static_cast<int>(groups.size()), C_API_DTYPE_INT32));
    if (!positions.empty()) checked(LGBM_DatasetSetField(handle, "position", positions.data(), n, C_API_DTYPE_INT32));
    const Dataset* data = reinterpret_cast<Dataset*>(handle);
    std::unique_ptr<ObjectiveFunction> objective(ObjectiveFunction::CreateObjectiveFunction(config.objective, config));
    objective->Init(data->metadata(), n);
    const std::string kind = c["kind"].string_value();
    A steps;
    if (kind == "dart") {
      DART dart; dart.Init(&config, data, objective.get(), {});
      for (int iter = 0; iter < c["num_steps"].int_value(); ++iter) {
        bool stopped = dart.TrainOneIter(nullptr, nullptr);
        steps.emplace_back(O{{"iteration", J(iter)}, {"drop_indices", J(dart.drop_index_)},
          {"tree_weights", J(dart.tree_weight_)}, {"sum_weight", J(dart.sum_weight_)}, {"stopped", J(stopped)}});
        if (stopped) break;
      }
    } else if (kind == "features") {
      ColSampler sampler(&config); sampler.SetTrainingData(data);
      Tree root(config.num_leaves, true, false);
      for (int iter = 0; iter < c["num_steps"].int_value(); ++iter) {
        // SetTrainingData already consumes the first tree sample.
        if (iter > 0) sampler.ResetByTree();
        auto bytree = sampler.is_feature_used_bytree();
        auto node0 = sampler.GetByNode(&root, 0), node1 = sampler.GetByNode(&root, 0);
        steps.emplace_back(O{{"iteration", J(iter)}, {"by_tree", J(values<int>(J(bytree)))},
          {"by_node", J(A{J(values<int>(J(node0))), J(values<int>(J(node1)))})}});
      }
    } else if (kind == "quantization") {
      auto gradients = values<score_t>(c["gradients"]), hessians = values<score_t>(c["hessians"]);
      GradientDiscretizer discretizer(config.num_grad_quant_bins, c["num_steps"].int_value(), config.seed,
                                     c["constant_hessian"].bool_value(), config.stochastic_rounding);
      discretizer.Init(n, config.num_leaves, features, data);
      for (int iter = 0; iter < c["num_steps"].int_value(); ++iter) {
        discretizer.DiscretizeGradients(n, gradients.data(), hessians.data());
        std::vector<int> packed;
        for (int i = 0; i < 2 * n; ++i) packed.push_back(discretizer.discretized_gradients_and_hessians()[i]);
        steps.emplace_back(O{{"iteration", J(iter)}, {"packed_hessian_gradient", J(packed)},
          {"grad_scale", J(discretizer.grad_scale())}, {"hess_scale", J(discretizer.hess_scale())}});
      }
    } else {
      std::unique_ptr<TreeLearner> learner(TreeLearner::CreateTreeLearner("serial", "cpu", &config, false));
      learner->Init(data, false);
      std::unique_ptr<SampleStrategy> sampler(SampleStrategy::CreateSampleStrategy(&config, data, objective.get(), config.num_class));
      sampler->UpdateObjectiveFunction(objective.get()); sampler->ResetSampleConfig(&config, true);
      auto scores = values<double>(c["scores"]);
      for (int iter = 0; iter < c["num_steps"].int_value(); ++iter) {
        std::vector<score_t> gradients(scores.size()), hessians(scores.size());
        objective->GetGradients(scores.data(), gradients.data(), hessians.data());
        O step{{"iteration", J(iter)}, {"input_gradients", J(gradients)}, {"input_hessians", J(hessians)}};
        sampler->Bagging(iter, learner.get(), gradients.data(), hessians.data());
        bool all_rows = config.data_sample_strategy == "goss" && iter < static_cast<int>(1.0f / config.learning_rate);
        std::vector<data_size_t> selected;
        if (!all_rows) selected.assign(sampler->bag_data_indices().begin(), sampler->bag_data_indices().begin() + sampler->bag_data_cnt());
        step["all_rows"] = J(all_rows); step["selected_rows"] = J(selected); step["selected_count"] = J(sampler->bag_data_cnt());
        step["gradients"] = J(gradients); step["hessians"] = J(hessians);
        if (config.bagging_by_query) step["selected_queries"] = J(std::vector<data_size_t>(sampler->sampled_query_indices(), sampler->sampled_query_indices() + sampler->num_sampled_queries()));
        steps.emplace_back(step);
      }
    }
    O result = c.object_items(); result["steps"] = J(steps);
    if (kind == "features") {
      std::vector<int> inner;
      for (int f = 0; f < features; ++f) inner.push_back(data->InnerFeatureIndex(f));
      result["eligible_features"] = J(data->ValidFeatureIndices());
      result["inner_feature_indices"] = J(inner);
    }
    cases.emplace_back(result);
    checked(LGBM_DatasetFree(handle));
  }
  save(argv[2], "sampling", cases);
}
