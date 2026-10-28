// Native-only fixture producer; shares output utilities with primitive_driver.cpp.
#include "driver_common.hpp"
#include <LightGBM/bin.h>
#include <LightGBM/config.h>
#include <LightGBM/dataset.h>
#include <LightGBM/metric.h>
#include <LightGBM/objective_function.h>
#include <limits>
#include <memory>
#include <sstream>
using namespace LightGBM;

template<class T> std::vector<T> values(const J& j) {
  std::vector<T> out;
  for (const auto& v : j.array_items()) out.push_back(static_cast<T>(v.number_value()));
  return out;
}
J scalar(double v) {
  if (std::isnan(v)) return J("NaN");
  if (std::isinf(v)) return J(v > 0 ? "Infinity" : "-Infinity");
  return J(v);
}
Config configuration(const J& params) {
  Config config;
  std::unordered_map<std::string, std::string> fields;
  for (const auto& p : params.object_items()) {
    if (p.second.is_array()) {
      std::string s;
      for (const auto& v : p.second.array_items()) { if (!s.empty()) s += ","; s += v.is_string() ? v.string_value() : v.dump(); }
      fields[p.first] = s;
    } else fields[p.first] = p.second.is_string() ? p.second.string_value() : p.second.dump();
  }
  config.Set(fields);
  return config;
}
void metadata_for(const J& c, Metadata* m) {
  auto labels = values<label_t>(c["labels"]);
  auto weights = values<label_t>(c["weights"]);
  auto groups = values<data_size_t>(c["groups"]);
  auto positions = values<data_size_t>(c["positions"]);
  m->Init(static_cast<data_size_t>(labels.size()), weights.empty() ? -1 : 0, -1);
  m->SetLabel(labels.data(), static_cast<data_size_t>(labels.size()));
  if (!weights.empty()) m->SetWeights(weights.data(), static_cast<data_size_t>(weights.size()));
  if (!groups.empty()) m->SetQuery(groups.data(), static_cast<data_size_t>(groups.size()));
  if (!positions.empty()) m->SetPosition(positions.data(), static_cast<data_size_t>(positions.size()));
}
J derivatives(ObjectiveFunction* objective, const std::vector<double>& scores) {
  std::vector<score_t> g(scores.size()), h(scores.size());
  objective->GetGradients(scores.data(), g.data(), h.data());
  return J(O{{"scores", J(scores)}, {"gradients", J(g)}, {"hessians", J(h)}});
}
std::vector<std::vector<double>> matrix(const J& j) {
  std::vector<std::vector<double>> out;
  for (const auto& row : j.array_items()) {
    auto v = values<double>(row);
    for (size_t i = 0; i < v.size(); ++i) if (row[i].is_null()) v[i] = std::numeric_limits<double>::quiet_NaN();
    out.push_back(v);
  }
  return out;
}
J shared_binning(const J& c) {
  auto rows = matrix(c["rows"]), valid_rows = matrix(c["validation_rows"]);
  auto forced = matrix(c["forced_upper_bounds"]);
  auto max_bins = values<int>(c["max_bin_by_feature"]);
  Config config;
  config.enable_bundle = false; config.feature_pre_filter = false; config.min_data_in_leaf = 1;
  config.min_data_in_bin = 1; config.max_bin = 15; config.max_bin_by_feature = max_bins;
  config.zero_as_missing = c["zero_as_missing"].bool_value();
  int ncol = static_cast<int>(rows[0].size()), n = static_cast<int>(rows.size());
  std::vector<std::vector<double>> samples(ncol);
  std::vector<std::vector<int>> indices(ncol);
  std::vector<std::unique_ptr<BinMapper>> mappers;
  std::vector<double*> sample_ptrs; std::vector<int*> index_ptrs; std::vector<int> counts;
  for (int f = 0; f < ncol; ++f) {
    for (int r = 0; r < n; ++r) if (std::isnan(rows[r][f]) || std::fabs(rows[r][f]) > kZeroThreshold) {
      samples[f].push_back(rows[r][f]); indices[f].push_back(r);
    }
    auto sample_copy = samples[f];
    std::unique_ptr<BinMapper> mapper(new BinMapper());
    mapper->FindBin(sample_copy.data(), static_cast<int>(sample_copy.size()), n, max_bins[f], 1, 1, false, BinType::NumericalBin, true, config.zero_as_missing, forced[f]);
    mappers.push_back(std::move(mapper)); sample_ptrs.push_back(samples[f].data()); index_ptrs.push_back(indices[f].data()); counts.push_back(static_cast<int>(samples[f].size()));
  }
  Dataset train(n);
  train.Construct(&mappers, ncol, forced, index_ptrs.data(), sample_ptrs.data(), counts.data(), ncol, n, config);
  for (int r = 0; r < n; ++r) train.PushOneRow(0, r, rows[r]);
  train.FinishLoad();
  Dataset valid(static_cast<data_size_t>(valid_rows.size())); valid.CreateValid(&train);
  for (size_t r = 0; r < valid_rows.size(); ++r) valid.PushOneRow(0, static_cast<data_size_t>(r), valid_rows[r]);
  valid.FinishLoad();
  auto bins = [ncol](const Dataset& ds, size_t count) {
    std::vector<std::unique_ptr<BinIterator>> iterators;
    for (int f = 0; f < ncol; ++f) iterators.emplace_back(ds.FeatureIterator(ds.InnerFeatureIndex(f)));
    A out;
    for (size_t r = 0; r < count; ++r) {
      A row; for (int f = 0; f < ncol; ++f) row.emplace_back(static_cast<int>(iterators[f]->Get(static_cast<data_size_t>(r))));
      out.emplace_back(row);
    }
    return J(out);
  };
  A bounds;
  for (int f = 0; f < ncol; ++f) {
    const auto* mapper = train.FeatureBinMapper(train.InnerFeatureIndex(f));
    std::vector<double> upper;
    for (int b = 0; b < mapper->num_bin(); ++b) upper.push_back(mapper->BinToValue(b));
    bounds.emplace_back(finite_values(upper, true));
  }
  if (!train.CheckAlign(valid)) throw std::runtime_error("validation bins not aligned");
  return J(O{{"training_bins", bins(train, rows.size())}, {"validation_bins", bins(valid, valid_rows.size())}, {"upper_bounds", J(bounds)}});
}
int main(int argc, char** argv) {
  if (argc != 3) return 2;
  static_assert(sizeof(score_t) == 4 && sizeof(label_t) == 4 && sizeof(hist_t) == 8);
  Log::ResetLogLevel(LogLevel::Fatal);
  std::ifstream input(argv[1]);
  std::stringstream buffer; buffer << input.rdbuf();
  std::string error;
  J request = J::parse(buffer.str(), &error);
  if (!error.empty()) throw std::runtime_error(error);
  A metadata_cases;
  for (const auto& c : request["dataset_metadata"].array_items()) {
    Metadata metadata; metadata_for(c, &metadata);
    auto initial = values<double>(c["initial_scores"]);
    if (!initial.empty()) metadata.SetInitScore(initial.data(), static_cast<data_size_t>(initial.size()));
    size_t n = c["labels"].array_items().size();
    O expected{{"labels", J(std::vector<label_t>(metadata.label(), metadata.label() + n))}};
    if (metadata.weights()) expected["weights"] = J(std::vector<label_t>(metadata.weights(), metadata.weights() + n));
    if (metadata.init_score()) expected["initial_scores"] = J(std::vector<double>(metadata.init_score(), metadata.init_score() + metadata.num_init_score()));
    if (metadata.query_boundaries()) expected["query_boundaries"] = J(std::vector<data_size_t>(metadata.query_boundaries(), metadata.query_boundaries() + metadata.num_queries() + 1));
    if (metadata.query_weights()) expected["query_weights"] = J(std::vector<label_t>(metadata.query_weights(), metadata.query_weights() + metadata.num_queries()));
    if (metadata.positions()) {
      expected["positions"] = J(std::vector<data_size_t>(metadata.positions(), metadata.positions() + n));
      expected["position_ids"] = J(std::vector<std::string>(metadata.position_ids(), metadata.position_ids() + metadata.num_position_ids()));
    }
    if (c["binning"].is_object()) expected["binning"] = shared_binning(c["binning"]);
    O out = c.object_items(); out["expected"] = J(expected); metadata_cases.emplace_back(out);
  }
  save(argv[2], "dataset_metadata", metadata_cases);
  A forced_cases;
  for (const auto& c : request["forced_bins"].array_items()) {
    auto sample = values<double>(c["sample_values"]);
    auto forced = values<double>(c["forced_upper_bounds"]);
    auto probes = values<double>(c["probe_values"]);
    for (size_t i = 0; i < probes.size(); ++i) if (c["probe_values"][i].is_null()) probes[i] = std::numeric_limits<double>::quiet_NaN();
    for (size_t i = 0; i < sample.size(); ++i) if (c["sample_values"][i].is_null()) sample[i] = std::numeric_limits<double>::quiet_NaN();
    const J& p = c["params"];
    BinMapper mapper;
    mapper.FindBin(sample.data(), static_cast<int>(sample.size()), c["total_sample_count"].int_value(), p["max_bin"].int_value(), p["min_data_in_bin"].int_value(), 1, false, BinType::NumericalBin, p["use_missing"].bool_value(), p["zero_as_missing"].bool_value(), forced);
    std::vector<double> bounds; A bins;
    for (int b = 0; b < mapper.num_bin(); ++b) bounds.push_back(mapper.BinToValue(b));
    for (double probe : probes) bins.emplace_back(static_cast<int>(mapper.ValueToBin(probe)));
    const char* missing = mapper.missing_type() == MissingType::NaN ? "NaN" : mapper.missing_type() == MissingType::Zero ? "Zero" : "None";
    O out = c.object_items();
    out["num_bin"] = J(mapper.num_bin()); out["upper_bounds"] = finite_values(bounds, true);
    out["missing_type"] = J(missing); out["default_bin"] = J(static_cast<int>(mapper.GetDefaultBin()));
    out["most_freq_bin"] = J(static_cast<int>(mapper.GetMostFreqBin())); out["value_bins"] = J(bins);
    forced_cases.emplace_back(out);
  }
  save(argv[2], "forced_bins", forced_cases);
  for (const std::string family : {"extended_objectives", "metrics"}) {
    A out;
    for (const auto& c : request[family].array_items()) {
      Config config = configuration(c["params"]);
      Metadata metadata; metadata_for(c, &metadata);
      auto labels = values<label_t>(c["labels"]);
      auto scores = values<double>(c["scores"]);
      std::unique_ptr<ObjectiveFunction> objective(ObjectiveFunction::CreateObjectiveFunction(c["objective"].string_value(), config));
      objective->Init(metadata, static_cast<data_size_t>(labels.size()));
      O result = c.object_items();
      if (c["record_objective_metadata"].bool_value()) {
        result.erase("record_objective_metadata");
        A need_train;
        for (int k = 0; k < config.num_class; ++k) need_train.emplace_back(objective->ClassNeedTrain(k));
        result["source_class_need_train"] = J(need_train);
        result["model_objective"] = J(objective->ToString());
      }
      if (family == "extended_objectives") {
        A initial;
        int classes = config.num_class;
        for (int k = 0; k < classes; ++k) initial.emplace_back(scalar(objective->BoostFromScore(k)));
        result["initial_scores"] = J(initial);
        J d = derivatives(objective.get(), scores);
        result["gradients"] = d["gradients"]; result["hessians"] = d["hessians"];
        if (c["sequential"].bool_value()) {
          result.erase("sequential");
          result["steps"] = J(A{derivatives(objective.get(), scores)});
        }
        if (objective->IsRenewTreeOutput()) {
          A renewals;
          for (auto rows : {std::vector<data_size_t>{0, 2, 4}, std::vector<data_size_t>{1, 3, 5}, std::vector<data_size_t>{0, 1, 2, 3, 4, 5}}) {
            std::function<double(const LightGBM::label_t*, int)> residual = [&scores](const LightGBM::label_t* label, int row) -> double { return static_cast<double>(label[row]) - scores[row]; };
            double renewed = objective->RenewTreeOutput(0.375, residual, rows.data(), nullptr, static_cast<data_size_t>(rows.size()));
            renewals.emplace_back(O{{"class", J(0)}, {"rows", J(rows)}, {"newton_output", J(0.375)}, {"expected_output", scalar(renewed)}});
          }
          result["renewals"] = J(renewals);
        }
      } else {
        std::unique_ptr<Metric> metric(Metric::CreateMetric(c["metric"].string_value(), config));
        metric->Init(metadata, static_cast<data_size_t>(labels.size()));
        auto evaluated = metric->Eval(scores.data(), objective.get());
        A results;
        for (size_t i = 0; i < evaluated.size(); ++i) results.emplace_back(O{{"name", J(metric->GetName()[i])}, {"value", scalar(evaluated[i])}, {"higher_is_better", J(metric->factor_to_bigger_better() > 0)}});
        result["results"] = J(results);
      }
      out.emplace_back(result);
    }
    save(argv[2], family, out);
  }
}
