"""Fixed CPU cases. Reference timings are not Rust performance comparisons."""
import argparse
import json
import platform
import resource
import statistics
import sys
import time

import numpy as np
from scipy import sparse

from common import COMMIT, ROOT, load_reference, save_json
from generate_predictions import BASE

CASES = ("dense", "sparse", "categorical", "multiclass", "ranking")
TRAIN = 8192
VALID = 2048
FEATURES = 16
ROUNDS = 50
BOUNDS = {"dense": ("rmse", "<=", 0.45), "sparse": ("rmse", "<=", 0.30),
          "categorical": ("binary_logloss", "<=", 0.15),
          "multiclass": ("multi_logloss", "<=", 0.45), "ranking": ("ndcg@5", ">=", 0.90)}


def contract():
    return dict(schema_version=1, upstream_commit=COMMIT, seed=0,
                generator="NumPy PCG64(0), pinned dependencies, tools/oracle/benchmark.py::data",
                train_rows=TRAIN, validation_rows=VALID, features=FEATURES, num_boost_round=ROUNDS,
                num_leaves=15, max_bin=63, learning_rate=0.1, num_threads=1,
                deterministic=True, force_col_wise=True, sparse_density=0.1, query_size=16,
                resource_limits=dict(seconds_per_case=300, peak_rss_mib=512),
                performance_budgets=dict(training_ratio=1.5, inference_ratio=1.25, peak_memory_ratio=1.5),
                cases=[dict(name=n, metric=BOUNDS[n][0], comparison=BOUNDS[n][1], quality_bound=BOUNDS[n][2],
                            reference_delta=0.01 if n == "ranking" else 0.02) for n in CASES])


def data(name):
    rng = np.random.Generator(np.random.PCG64(0))
    x = rng.uniform(-1., 1., size=(TRAIN + VALID, FEATURES))
    if name == "categorical":
        x[:, 0] = rng.integers(0, 8, len(x))
        y = (x[:, 0].astype(np.int32) % 4 == 0).astype(np.float32)
    elif name == "multiclass":
        y = x[:, :3].argmax(axis=1).astype(np.float32)
    elif name == "ranking":
        y = np.minimum(3, ((x[:, 0] + 1.) * 2.).astype(np.int32)).astype(np.float32)
    elif name == "sparse":
        x *= rng.random(x.shape) < 0.1
        y = (2. * x[:, 0] - 1.5 * x[:, 1]).astype(np.float32)
        x = sparse.csr_matrix(x)
        x.sort_indices()
    else:
        y = (2. * x[:, 0] - 1.5 * x[:, 1] + 0.5 * x[:, 2] * x[:, 3]).astype(np.float32)
    return x[:TRAIN], y[:TRAIN], x[TRAIN:], y[TRAIN:]


def quality(name, labels, predictions):
    if name in ("dense", "sparse"):
        return float(np.sqrt(np.mean((labels - predictions) ** 2)))
    if name == "categorical":
        p = np.clip(predictions, 1e-15, 1 - 1e-15)
        return float(-np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p)))
    if name == "multiclass":
        return float(-np.mean(np.log(np.maximum(predictions[np.arange(len(labels)), labels.astype(int)], 1e-15))))
    dcg = []
    discount = 1. / np.log2(np.arange(2, 7))
    for y, p in zip(labels.reshape(-1, 16), predictions.reshape(-1, 16)):
        ideal = np.sort(y)[::-1][:5]
        selected = y[np.argsort(-p, kind="stable")[:5]]
        denominator = np.sum((2. ** ideal - 1.) * discount)
        dcg.append(float(np.sum((2. ** selected - 1.) * discount) / denominator) if denominator else 1.)
    return float(np.mean(dcg))


def case(lgb, name):
    x, y, valid_x, valid_y = data(name)
    params = dict(BASE)
    params.update(num_leaves=15, max_bin=63, min_data_in_leaf=20, min_data_in_bin=3,
                  learning_rate=0.1, objective={"categorical": "binary", "multiclass": "multiclass",
                  "ranking": "lambdarank"}.get(name, "regression"), metric="None")
    if name == "multiclass":
        params["num_class"] = 3
    groups = [16] * (TRAIN // 16) if name == "ranking" else None
    columns = [0] if name == "categorical" else "auto"
    construction, training, prediction, metrics = [], [], [], []
    start = time.perf_counter()
    for repetition in range(6):  # one warm-up, five measurements
        t = time.perf_counter()
        ds = lgb.Dataset(x, label=y, group=groups, categorical_feature=columns, params=params)
        ds.construct()
        built = time.perf_counter()
        model = lgb.train(params, ds, num_boost_round=ROUNDS)
        trained = time.perf_counter()
        pred = model.predict(valid_x, num_threads=1)
        predicted = time.perf_counter()
        score = quality(name, valid_y, pred)
        metric, comparison, bound = BOUNDS[name]
        assert score <= bound if comparison == "<=" else score >= bound, (name, metric, score, bound)
        if repetition:
            construction.append(built - t)
            training.append(trained - built)
            prediction.append(predicted - trained)
            metrics.append(score)
        assert time.perf_counter() - start < 300, "per-case time limit exceeded"
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak_bytes = peak if sys.platform == "darwin" else peak * 1024
        assert peak_bytes <= 512 * 1024 * 1024, "512 MiB process limit exceeded"
    assert len(set(metrics)) == 1, "reference quality was not repeatable"
    return dict(name=name, params=params, train_rows=TRAIN, validation_rows=VALID, columns=FEATURES,
                nonzero_training_values=int(x.nnz if sparse.issparse(x) else np.count_nonzero(x)),
                metric=metric, quality=metrics, dataset_construction_seconds=construction,
                training_seconds=training, inference_seconds=prediction,
                median_dataset_construction_seconds=statistics.median(construction),
                median_training_seconds=statistics.median(training), median_inference_seconds=statistics.median(prediction))


def export_inputs():
    baseline = json.loads((ROOT / "tests/fixtures/reference-baseline.json").read_text())
    for name in ("dense", "sparse"):
        x, y, valid_x, valid_y = data(name)
        native = next(c for c in baseline["cases"] if c["name"] == name)
        output = dict(schema_version=1, upstream_commit=COMMIT, name=name, rows=TRAIN, features=FEATURES,
                      validation_rows=VALID, num_boost_round=ROUNDS,
                      train_values=(x.toarray() if sparse.issparse(x) else x).ravel().tolist(),
                      train_labels=y.tolist(),
                      validation_values=(valid_x.toarray() if sparse.issparse(valid_x) else valid_x).ravel().tolist(),
                      validation_labels=valid_y.tolist(), params=native["params"],
                      reference_expected_quality=native["quality"][0], metric=native["metric"],
                      reference_median_training_seconds=native["median_training_seconds"],
                      reference_median_inference_seconds=native["median_inference_seconds"])
        if sparse.issparse(x):
            def csr(matrix):
                return dict(indptr=matrix.indptr.tolist(), indices=matrix.indices.tolist(), values=matrix.data.tolist())
            output["train_csr"] = csr(x)
            output["validation_csr"] = csr(valid_x)
        path = ROOT / f"target/oracle/benchmark-input-{name}.json"
        save_json(path, output)
        print(f"Exported exact {name} benchmark inputs to {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract-only", action="store_true")
    parser.add_argument("--export-inputs", action="store_true", help="export exact dense/sparse inputs under ignored target/oracle")
    args = parser.parse_args()
    if args.export_inputs:
        export_inputs()
        return
    save_json(ROOT / "tests/fixtures/benchmark-contract.json", contract())
    if args.contract_only:
        return
    lgb, identity = load_reference()
    output = dict(schema_version=1, upstream_commit=COMMIT, host=platform.platform(), machine=platform.machine(),
                  library=identity["library"], compiler=identity["compiler"], warmup_runs=1, measured_runs=5,
                  memory="Not measured at a matched native boundary; memory budget remains open",
                  cases=[case(lgb, name) for name in CASES])
    save_json(ROOT / "tests/fixtures/reference-baseline.json", output)
    for c in output["cases"]:
        print(c["name"], c["metric"], c["quality"][0], "median train", c["median_training_seconds"],
              "median predict", c["median_inference_seconds"])


if __name__ == "__main__":
    main()
