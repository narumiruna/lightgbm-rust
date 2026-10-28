"""Pinned native empty-query evidence. Never execute empty-query LambdaRank.

Run: uv run --project tools/oracle python tools/oracle/generate_empty_queries.py
The embedded driver uses native factories, not a Python numerical substitute.
"""
import json
import math
import shlex
import subprocess

from common import COMMIT, OUT, ROOT, SOURCE, load_reference, save_json, sha

LIBRARY_SHA = '9ed1d46d2d60bac2966fd03bd81e7d192025804f6a3e3ef24c3aa483d0ec7c1e'
DRIVER = r'''
#include "driver_common.hpp"
#include <LightGBM/config.h>
#include <LightGBM/dataset.h>
#include <LightGBM/metric.h>
#include <LightGBM/objective_function.h>
#include <memory>
#include <sstream>
using namespace LightGBM;
template<class T> std::vector<T> values(const J& j) {
  std::vector<T> out;
  for (const auto& v : j.array_items()) out.push_back(static_cast<T>(v.number_value()));
  return out;
}
int main(int argc, char** argv) {
  if (argc != 3) return 2;
  static_assert(sizeof(score_t) == 4 && sizeof(label_t) == 4);
  std::ifstream input(argv[1]); std::stringstream buffer; buffer << input.rdbuf();
  std::string error; auto request = J::parse(buffer.str(), &error);
  if (!error.empty()) throw std::runtime_error(error);
  A output;
  for (const auto& c : request["cases"].array_items()) {
    Config config; config.objective = "rank_xendcg"; config.num_threads = 1;
    config.objective_seed = 7; config.verbosity = -1;
    auto labels = values<label_t>(c["labels"]);
    auto groups = values<data_size_t>(c["groups"]);
    auto positions = values<data_size_t>(c["positions"]);
    Metadata metadata; metadata.Init(static_cast<data_size_t>(labels.size()), -1, -1);
    metadata.SetLabel(labels.data(), static_cast<data_size_t>(labels.size()));
    metadata.SetQuery(groups.data(), static_cast<data_size_t>(groups.size()));
    if (!positions.empty()) metadata.SetPosition(positions.data(), static_cast<data_size_t>(positions.size()));
    std::unique_ptr<ObjectiveFunction> objective(ObjectiveFunction::CreateObjectiveFunction("rank_xendcg", config));
    objective->Init(metadata, static_cast<data_size_t>(labels.size()));
    O out = c.object_items();
    out["boundaries"] = J(std::vector<data_size_t>(metadata.query_boundaries(), metadata.query_boundaries() + metadata.num_queries() + 1));
    std::vector<score_t> g(labels.size(), 0), h(labels.size(), 0);
    A steps;
    for (const auto& step : c["steps"].array_items()) {
      auto scores = values<double>(step["scores"]);
      if (step["queries"].is_array()) {
        auto queries = values<data_size_t>(step["queries"]);
        objective->GetGradientsWithSampledQueries(scores.data(), static_cast<data_size_t>(queries.size()), queries.data(), g.data(), h.data());
      } else objective->GetGradients(scores.data(), g.data(), h.data());
      O result = step.object_items(); result["gradients"] = J(g); result["hessians"] = J(h);
      A metrics;
      for (const std::string name : {"ndcg", "map"}) {
        std::unique_ptr<Metric> metric(Metric::CreateMetric(name, config));
        metric->Init(metadata, static_cast<data_size_t>(labels.size()));
        auto evaluated = metric->Eval(scores.data(), objective.get());
        for (size_t i = 0; i < evaluated.size(); ++i)
          metrics.emplace_back(O{{"name", J(metric->GetName()[i])}, {"value", J(evaluated[i])}});
      }
      result["metrics"] = J(metrics); steps.emplace_back(result);
    }
    out["steps"] = J(steps); output.emplace_back(out);
  }
  save(argv[2], "native", output);
}
'''


def specifications():
    definitions = [('nonempty', [2, 2]), ('leading', [0, 2, 2]),
                   ('middle', [2, 0, 2]), ('trailing', [2, 2, 0]),
                   ('repeated', [0, 0, 2, 0, 0, 2, 0, 0]),
                   ('short', [0, 1, 0, 2, 1, 0]), ('one_row', [0, 1, 0, 0])]
    cases = []
    for name, groups in definitions:
        n = sum(groups)
        scores = [0.4, -0.2, 0.1, 0.8][:n]
        for positions in [False, True]:
            queries = [i for i in range(len(groups)) if i % 2 == 0]
            case = dict(name=name + ('_positions' if positions else ''), groups=groups,
                        labels=[0.0, 1.0, 1.0, 0.0][:n],
                        steps=[dict(scores=scores), dict(scores=list(reversed(scores))),
                               dict(scores=scores, queries=queries), dict(scores=scores)])
            if positions:
                case['positions'] = [9, -7, 9, -7][:n]
            cases.append(case)

    return cases


NAMES = {c['name'] for c in specifications()}


def generate(fixtures=ROOT / 'tests/fixtures'):
    lgb, identity = load_reference()
    assert COMMIT == '5d84f3f222b41409a6983d2ac90513ad9702fa9e'
    assert identity['library']['loaded_sha256'] == LIBRARY_SHA
    work = OUT / 'empty-queries'
    work.mkdir(parents=True, exist_ok=True)
    source = work / 'driver.cpp'
    source.write_text(DRIVER)
    build = OUT / 'build'
    command = shlex.split(json.loads((build / 'compile_commands.json').read_text())[0]['command'])
    command[command.index('-o') + 1] = str(work / 'driver.o')
    command[command.index('-c') + 1] = str(source)
    command.extend([f'-I{ROOT / "tools/oracle"}', f'-I{SOURCE}', f'-I{OUT}'])
    subprocess.run(command, cwd=build, check=True, timeout=300)
    objects = sorted((build / 'CMakeFiles/lightgbm_objs.dir').rglob('*.o'))
    link = [*command[:command.index('-o')], str(work / 'driver.o'), *map(str, objects),
            str(build / 'external_libs/nanoarrow/libnanoarrow_static.a'), '-o', str(work / 'driver')]
    subprocess.run(link, check=True, timeout=300)
    cases = specifications()
    request = work / 'request.json'
    save_json(request, dict(cases=cases))
    subprocess.run([str(work / 'driver'), str(request), str(work)], check=True, timeout=300)
    native = json.loads((work / 'native.json').read_text())['cases']
    # NumPy supplies only the matrix storage accepted by the Python adapter.
    # All training, derivatives, metric values, and predictions are native.
    import numpy as np
    params = dict(objective='rank_xendcg', objective_seed=7, metric=['ndcg', 'map'],
                  num_threads=1, verbosity=-1, min_data_in_leaf=1, min_data_in_bin=1,
                  feature_pre_filter=False, enable_bundle=False, num_leaves=3,
                  learning_rate=0.1, deterministic=True, force_col_wise=True)
    for case in native:
        n = len(case['labels'])
        x = np.arange(n, dtype=np.float64).reshape(n, 1)
        modes = [('training', params)]
        # Native bag_query_indices_ and RNG streams are sized by num_data_,
        # not num_queries_. Do not execute the unsafe queries > rows case.
        if len(case['groups']) <= n:
            modes.append(('query_bagging', params | dict(
                bagging_by_query=True, bagging_freq=1, bagging_fraction=0.75, bagging_seed=3)))
        for field, training_params in modes:
            data = lgb.Dataset(x, label=case['labels'], group=case['groups'],
                               position=case.get('positions'), params=training_params,
                               free_raw_data=False)
            history = {}
            booster = lgb.train(training_params, data, num_boost_round=3, valid_sets=[data],
                                callbacks=[lgb.record_evaluation(history)])
            predictions = [booster.predict(x, raw_score=True, num_iteration=i).tolist()
                           for i in range(1, 4)]
            assert all(math.isfinite(v) for row in predictions for v in row)
            assert all(math.isfinite(v) for values in history['training'].values() for v in values)
            case[field] = dict(params=training_params, rounds=3, predictions=predictions,
                               metrics=history['training'], trees=booster.num_trees())
    fixture = dict(schema_version=1, upstream_commit=COMMIT, loaded_library_sha256=LIBRARY_SHA,
                   native_driver_sha256=sha(work / 'driver'), native_driver_source_sha256=sha(source),
                   generator_sha256=sha(ROOT / 'tools/oracle/generate_empty_queries.py'),
                   source_sha256={p: sha(SOURCE / p) for p in [
                       'src/io/metadata.cpp', 'src/objective/rank_objective.hpp',
                       'src/metric/rank_metric.hpp', 'src/metric/map_metric.hpp',
                       'src/boosting/bagging.hpp']},
                   restrictions={
                       'lambdarank_empty': 'Not executed: rank_objective.hpp:201 indexes sorted_idx[0] when cnt=0 (undefined behavior).',
                       'weighted_empty': 'Not executed: metadata.cpp:852-864 divides query weight sum by zero; no finite query-weight semantics.',
                       'query_bagging_queries_exceed_rows': 'Not executed: bagging.hpp:176-181 sizes bag_query_indices_ and bagging_rands_ by num_data_ but indexes them with query indexes.',
                       'row_subset_empty': 'Not executed: metadata.cpp:144-161 indexes data_idx + len - 1 for zero-length groups; identity subsets preserve all query entries in Rust.'},
                   cases=native)
    path = fixtures / 'empty_queries.json'
    save_json(path, fixture)
    print(f'Generated {len(native)} native factory and real-training cases: {path}')


if __name__ == '__main__':
    generate()
