"""Load Rust-exported text models in the verified native reference (developer-only)."""
import argparse
from pathlib import Path

import numpy as np

from common import COMMIT, ROOT, load_reference, save_json, sha
from check import PREDICTION_CASES, read_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_directory", type=Path)
    args = parser.parse_args()
    lgb, identity = load_reference()
    fixtures = read_json(ROOT / "tests/fixtures/prediction.json")
    paths = list(args.model_directory.glob("*.txt"))
    assert {p.stem for p in paths} == PREDICTION_CASES, "missing or unexpected Rust-exported model"
    results = []
    for case in fixtures["cases"]:
        model_path = args.model_directory / (case["name"] + ".txt")
        model = lgb.Booster(model_file=str(model_path))
        rows = np.array([[np.nan if v is None else v for v in row] for row in case["rows"]], dtype=np.float64)
        maximum_errors = {}
        for kind, kwargs in (("raw", {"raw_score": True}), ("transformed", {}), ("leaves", {"pred_leaf": True})):
            actual = np.asarray(model.predict(rows, num_threads=1, **kwargs)).reshape(len(rows), -1)
            expected = np.asarray(case[kind])
            assert actual.shape == expected.shape, (case["name"], kind, actual.shape, expected.shape)
            assert np.isfinite(actual).all()
            if kind == "leaves":
                assert np.array_equal(actual, expected), (case["name"], kind)
            else:
                assert np.all(np.abs(actual - expected) <= 1e-12 + 1e-10 * np.abs(expected)), (case["name"], kind)
                maximum_errors[kind] = float(np.max(np.abs(actual - expected)))
        results.append(dict(name=case["name"], rust_model_sha256=sha(model_path),
                            reference_model_sha256=sha(ROOT / "tests/fixtures" / case["model_file"]),
                            rows=len(rows), maximum_absolute_error=maximum_errors, leaf_indexes_exact=True))
    save_json(ROOT / "tests/fixtures/rust-model-roundtrip.json",
              dict(schema_version=1, upstream_commit=COMMIT, library=identity["library"],
                   rust_model_directory=str(args.model_directory.resolve()), cases=results))
    print(f"Native reference loaded and checked all {len(results)} Rust-exported models: scores within bounds; leaf indexes exact")


if __name__ == "__main__":
    main()
