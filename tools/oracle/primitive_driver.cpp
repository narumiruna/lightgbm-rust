// Reference-only driver. See THIRD_PARTY_NOTICES.md for upstream MIT notices.
// Calls the actual pinned kernels; this is not an independent reimplementation.
#include <LightGBM/bin.h>
#include <LightGBM/config.h>
#include <LightGBM/dataset.h>
#include <LightGBM/utils/json11.h>
#include <LightGBM/utils/random.h>
#include "src/objective/regression_objective.hpp"
#include "src/objective/binary_objective.hpp"
#include "src/treelearner/feature_histogram.hpp"

#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

using namespace LightGBM;
#include "driver_common.hpp"

A random_cases() {
  A cases;
  for (int seed : {0, 1, 42}) {
    Random s(seed), i(seed), f(seed), sample(seed), dense(seed), mixed(seed);
    A shorts, ints, floats;
    for (int k = 0; k < 16; ++k) {
      shorts.emplace_back(s.NextShort(0, 1000));
      ints.emplace_back(i.NextInt(0, 1000));
      floats.emplace_back(static_cast<double>(f.NextFloat()));
    }
    A mixed_values;
    for (int k = 0; k < 4; ++k) {
      mixed_values.emplace_back(O{{"method", J("NextShort")}, {"value", J(mixed.NextShort(0, 1000))}});
      mixed_values.emplace_back(O{{"method", J("NextFloat")}, {"value", J(static_cast<double>(mixed.NextFloat()))}});
      mixed_values.emplace_back(O{{"method", J("NextInt")}, {"value", J(mixed.NextInt(0, 1000))}});
    }
    cases.emplace_back(O{{"name", J("seed" + std::to_string(seed))}, {"seed", J(seed)},
                         {"lower", J(0)}, {"upper", J(1000)}, {"next_short", J(shorts)},
                         {"next_int", J(ints)}, {"next_float", J(floats)},
                         {"sample_n", J(20)}, {"sample_k", J(6)}, {"sample", J(sample.Sample(20, 6))},
                         {"sample_dense", J(O{{"n", J(20)}, {"k", J(12)}, {"indices", J(dense.Sample(20, 12))}})},
                         {"mixed", J(mixed_values)}});
  }
  return cases;
}

A objective_cases() {
  A cases;
  for (bool binary : {false, true}) for (bool weighted : {false, true}) {
    Config config;
    config.deterministic = true;
    config.sigmoid = 1.;
    std::vector<label_t> labels = binary ? std::vector<label_t>{0, 1, 0, 1, 1, 0} :
                                         std::vector<label_t>{-2.25f, -0.1f, 0.f, 1.1f, 3.75f, 7.f};
    std::vector<label_t> weights{1.f, 0.1f, 2.f, 0.5f, 3.f, 1.25f};
    std::vector<double> scores{-3., -0.125, 0., 0.3, 2., 10.};
    Metadata metadata;
    metadata.Init(6, weighted ? 0 : -1, -1);
    metadata.SetLabel(labels.data(), 6);
    if (weighted) metadata.SetWeights(weights.data(), 6);
    std::unique_ptr<ObjectiveFunction> objective(binary ? static_cast<ObjectiveFunction*>(new BinaryLogloss(config)) :
                                                        static_cast<ObjectiveFunction*>(new RegressionL2loss(config)));
    objective->Init(metadata, 6);
    std::vector<score_t> gradients(6), hessians(6);
    objective->GetGradients(scores.data(), gradients.data(), hessians.data());
    O c{{"name", J(std::string(binary ? "binary" : "l2") + (weighted ? "_weighted" : ""))},
        {"objective", J(binary ? "binary" : "regression")}, {"labels", J(labels)},
        {"scores", J(scores)}, {"gradients", J(gradients)}, {"hessians", J(hessians)},
        {"init_score", J(objective->BoostFromScore(0))},
        {"params", J(O{{"deterministic", J(true)}, {"sigmoid", J(1.)},
                         {"is_unbalance", J(false)}, {"scale_pos_weight", J(1.)}, {"reg_sqrt", J(false)}})}};
    if (weighted) c["weights"] = J(weights);
    cases.emplace_back(c);
  }
  return cases;
}

A bin_cases() {
  A cases;
  const double nan = std::numeric_limits<double>::quiet_NaN();
  for (std::string name : {"numeric", "nan_missing", "zero_missing", "constant", "max_bin4",
                           "zero_separated", "missing_disabled", "max_bin2", "max_bin3",
                           "min_data_in_bin3", "adjacent_doubles", "categorical_ties",
                           "categorical_missing", "categorical_rare"}) {
    bool categorical = name.find("categorical_") == 0;
    bool missing = name == "nan_missing" || name == "zero_missing" || name == "missing_disabled" || name == "categorical_missing";
    bool zero_missing = name == "zero_missing";
    bool use_missing = name != "missing_disabled";
    std::vector<double> sample{-3., -2., -1., 1., 2., 3., 4., 5.};
    size_t total = 10;  // two implicit zeros are not passed to FindBin
    if (missing) { sample.push_back(nan); ++total; }
    if (name == "constant") { sample = {2., 2., 2., 2.}; total = 4; }
    int max_bin = name == "max_bin4" ? 4 : name == "max_bin2" ? 2 : name == "max_bin3" ? 3 : 8;
    int min_data = name == "min_data_in_bin3" ? 3 : 1;
    if (name == "zero_separated") { sample = {-2., -1., -1e-36, 1e-36, 1., 2.}; total = 8; }
    if (name == "adjacent_doubles") { sample = {1., std::nextafter(1., 2.), 2., std::nextafter(2., 3.), 3., 4.}; total = 6; }
    if (name == "categorical_ties") { sample = {1., 2., 3., 1., 2., 3.}; total = 8; }
    if (name == "categorical_missing") { sample = {-2., -1., 1., 1., 2., 2., 3., nan}; total = 10; }
    if (name == "categorical_rare") { sample = {1., 1., 1., 2., 2., 3., 4., 5., 6.}; total = 12; max_bin = 3; }
    auto sorted = sample;
    BinMapper mapper;
    mapper.FindBin(sorted.data(), static_cast<int>(sorted.size()), total, max_bin, min_data, 1,
                   false, categorical ? BinType::CategoricalBin : BinType::NumericalBin, use_missing, zero_missing, {});
    std::vector<double> probes{-4., -3., -2., -1., -1e-36, 0., 1e-36, 0.5, 1., 2., 3., 6., nan};
    if (name == "adjacent_doubles") probes = {0., 1., std::nextafter(1., 2.), 2., std::nextafter(2., 3.), 3., 4., nan};
    if (categorical) probes = {-3., -1., -0.5, 0., 0.9, 1., 1.9, 2., 3., 4., 5., 6., 8., nan};
    std::vector<double> bounds;
    A bins;
    for (int b = 0; b < mapper.num_bin(); ++b) bounds.push_back(mapper.BinToValue(b));
    for (double p : probes) bins.emplace_back(static_cast<int>(mapper.ValueToBin(p)));
    const char* type = mapper.missing_type() == MissingType::NaN ? "NaN" :
                       mapper.missing_type() == MissingType::Zero ? "Zero" : "None";
    cases.emplace_back(O{{"name", J(name)}, {"sample_values", finite_values(sample)},
      {"total_sample_count", J(static_cast<int>(total))}, {"probe_values", finite_values(probes)},
      {"is_categorical", J(categorical)},
      {"params", J(O{{"max_bin", J(max_bin)}, {"min_data_in_bin", J(min_data)}, {"min_split_data", J(1)},
                       {"pre_filter", J(false)}, {"use_missing", J(use_missing)}, {"zero_as_missing", J(zero_missing)},
                       {"bin_type", J(categorical ? "CategoricalBin" : "NumericalBin")}})},
      {"num_bin", J(mapper.num_bin())}, {"upper_bounds", finite_values(bounds, true)},
      {"missing_type", J(type)}, {"default_bin", J(static_cast<int>(mapper.GetDefaultBin()))},
      {"most_freq_bin", J(static_cast<int>(mapper.GetMostFreqBin()))}, {"value_bins", J(bins)}});
  }
  return cases;
}

A histogram_cases() {
  A cases;
  for (double lambda : {0., 1.}) {
    Config config;
    config.min_data_in_leaf = 1;
    config.min_sum_hessian_in_leaf = 0.001;
    config.lambda_l2 = lambda;
    std::vector<int> bins{0, 0, 1, 1, 2, 2, 3, 3};
    std::vector<score_t> gradients{-3.f, -2.f, -1.f, 0.f, 1.f, 2.f, 3.f, 4.f};
    std::vector<score_t> hessians(8, 1.f);
    std::unique_ptr<Bin> bin(Bin::CreateDenseBin(8, 4));
    for (int row = 0; row < 8; ++row) bin->Push(0, row, bins[row]);
    bin->FinishLoad();
    std::vector<hist_t> histogram(8 + 32, 0.);
    bin->ConstructHistogram(0, 8, gradients.data(), hessians.data(), histogram.data());
    FeatureMetainfo meta;
    meta.num_bin = 4;
    meta.missing_type = MissingType::None;
    meta.default_bin = 0;
    meta.bin_type = BinType::NumericalBin;
    meta.config = &config;
    meta.rand = Random(0);
    FeatureHistogram feature;
    feature.Init(histogram.data(), &meta);
    BasicConstraintEntry constraints;
    SplitInfo split;
    feature.FindBestThreshold(4., 8., 8, &constraints, 0., &split);
    A values;
    for (int b = 0; b < 4; ++b) values.emplace_back(A{J(histogram[2*b]), J(histogram[2*b+1])});
    cases.emplace_back(O{{"name", J(lambda == 0 ? "numeric_split" : "numeric_split_l2")},
      {"bins", J(bins)}, {"gradients", J(gradients)}, {"hessians", J(hessians)}, {"histogram", J(values)},
      {"params", J(O{{"min_data_in_leaf", J(1)}, {"min_sum_hessian_in_leaf", J(0.001)},
                      {"lambda_l1", J(0.)}, {"lambda_l2", J(lambda)}, {"min_gain_to_split", J(0.)},
                      {"max_delta_step", J(0.)}, {"path_smooth", J(0.)}})},
      {"expected_split", J(O{{"threshold", J(static_cast<int>(split.threshold))},
        {"left_count", J(split.left_count)}, {"right_count", J(split.right_count)},
        {"left_output", J(split.left_output)}, {"right_output", J(split.right_output)}, {"gain", J(split.gain)},
        {"left_sum_gradient", J(split.left_sum_gradient)}, {"left_sum_hessian", J(split.left_sum_hessian)},
        {"right_sum_gradient", J(split.right_sum_gradient)}, {"right_sum_hessian", J(split.right_sum_hessian)},
        {"default_left", J(split.default_left)}})}});
  }
  return cases;
}

int main(int argc, char** argv) {
  if (argc != 2) return 2;
  static_assert(sizeof(score_t) == 4 && sizeof(label_t) == 4 && sizeof(hist_t) == 8);
  Log::ResetLogLevel(LogLevel::Fatal);
  save(argv[1], "rng", random_cases());
  save(argv[1], "objectives", objective_cases());
  save(argv[1], "bins", bin_cases());
  save(argv[1], "histograms", histogram_cases());
  Config defaults;
  O declared;
  #include "parameter_defaults.inc"
  A missing_defaults;
  for (const auto& entry : Config::parameter2aliases()) {
    if (!declared.count(entry.first)) missing_defaults.emplace_back(entry.first);
  }
  if (!missing_defaults.empty()) throw std::runtime_error("missing parameter defaults: " + J(missing_defaults).dump());
  std::ofstream inventory(std::string(argv[1]) + "/parameters.json");
  inventory << J(O{{"schema_version", J(1)}, {"upstream_commit", J(commit)},
    {"aliases", J(Config::parameter2aliases())}, {"types", J(Config::ParameterTypes())},
    {"declared_defaults", J(declared)},
    {"default_notes", J("Default-constructed Config fields. task uses the TaskType enum integer (kTrain=0). Empty vectors can be expanded when Config::Set is called.")},
    {"serialized_defaults", J(defaults.ToString())}}).dump() << '\n';
}
