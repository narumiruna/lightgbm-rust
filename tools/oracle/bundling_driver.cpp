// Development-only native grouping evidence. Pinned LightGBM MIT sources.
#include "driver_common.hpp"
#include <LightGBM/dataset.h>
#include <LightGBM/dataset_loader.h>
#include <iostream>
#include <iterator>
using namespace LightGBM;
int main() {
  Log::ResetLogLevel(LogLevel::Fatal);
  std::string error;
  auto input = J::parse(std::string(std::istreambuf_iterator<char>(std::cin), {}), &error);
  if (!error.empty()) throw std::runtime_error(error);
  A cases;
  for (const auto& c : input.array_items()) {
    const int n = c["rows"].array_items().size();
    const int p = c["rows"][0].array_items().size();
    Config config;
    config.Set(Config::Str2Map(c["parameters"].string_value().c_str()));
    std::vector<std::vector<double>> vals(p);
    std::vector<std::vector<int>> inds(p);
    for (int i = 0; i < n; ++i) for (int j = 0; j < p; ++j) {
      const auto v = c["rows"][i][j];
      double x = v.is_null() ? std::numeric_limits<double>::quiet_NaN() : v.number_value();
      if (std::fabs(x) > kZeroThreshold || std::isnan(x)) { vals[j].push_back(x); inds[j].push_back(i); }
    }
    std::vector<double*> vp; std::vector<int*> ip; std::vector<int> sizes;
    for (int j = 0; j < p; ++j) { vp.push_back(vals[j].data()); ip.push_back(inds[j].data()); sizes.push_back(vals[j].size()); }
    DatasetLoader loader(config, nullptr, 1, nullptr);
    std::unique_ptr<Dataset> d(loader.ConstructFromSampleData(vp.data(), ip.data(), p, sizes.data(), n, n, n));
    // FindBin sorts values in place, as does the actual native loader. Push the original matrix.
    for (int i = 0; i < n; ++i) {
      std::vector<double> row;
      for (int j = 0; j < p; ++j) row.push_back(c["rows"][i][j].is_null() ? std::numeric_limits<double>::quiet_NaN() : c["rows"][i][j].number_value());
      d->PushOneRow(0, i, row);
    }
    d->FinishLoad();
    A mapping, groups, maps, bins;
    for (int j = 0; j < p; ++j) {
      const int f = d->InnerFeatureIndex(j); mapping.emplace_back(f);
      if (f < 0) { maps.emplace_back(nullptr); continue; }
      const auto* m = d->FeatureBinMapper(f);
      maps.emplace_back(O{{"default_bin", J(int(m->GetDefaultBin()))}, {"most_freq_bin", J(int(m->GetMostFreqBin()))}, {"missing_type", J(int(m->missing_type()))}, {"num_bins", J(m->num_bin())}, {"group", J(d->Feature2Group(f))}});
    }
    for (int g = 0; g < d->num_feature_groups(); ++g) {
      A fs; int conflicts = 0;
      for (int f = 0; f < d->num_features(); ++f) if (d->Feature2Group(f) == g) fs.emplace_back(d->RealFeatureIndex(f));
      for (int i = 0; i < n; ++i) {
        int active = 0;
        for (const auto& j : fs) {
          const int f = d->InnerFeatureIndex(j.int_value());
          std::unique_ptr<BinIterator> it(d->FeatureIterator(f));
          active += it->Get(i) != d->FeatureBinMapper(f)->GetMostFreqBin();
        }
        conflicts += std::max(0, active - 1);
      }
      groups.emplace_back(O{{"features", J(fs)}, {"multi_value", J(bool(d->IsMultiGroup(g)))}, {"decoded_conflicts", J(conflicts)}});
    }
    for (int i = 0; i < n; ++i) {
      A row;
      for (int j = 0; j < p; ++j) {
        int f = d->InnerFeatureIndex(j);
        if (f < 0) row.emplace_back(nullptr);
        else { std::unique_ptr<BinIterator> it(d->FeatureIterator(f)); row.emplace_back(int(it->Get(i))); }
      }
      bins.emplace_back(row);
    }
    auto o = c.object_items(); o["inner_features"] = J(mapping); o["groups"] = J(groups); o["maps"] = J(maps); o["bins"] = J(bins);
    cases.emplace_back(o);
  }
  std::cout << J(cases).dump() << '\n';
}
