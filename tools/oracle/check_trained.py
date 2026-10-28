"""Reload genuinely Rust-trained models through the pinned native reference."""
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

from common import COMMIT, OUT, ROOT, load_reference, save_json, sha


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: check_trained.py MODEL_DIRECTORY RUST_BINARY")
    directory = Path(sys.argv[1]).resolve()
    binary = Path(sys.argv[2]).resolve()
    lgb, identity = load_reference()
    rows = np.array([[-3.], [-2.], [-1.], [0.], [1.], [2.], [3.], [4.]], dtype=np.float64)
    cases = []
    with tempfile.TemporaryDirectory(prefix="rust-trained-", dir=OUT) as temporary:
        input_path = Path(temporary) / "features.csv"
        input_path.write_text("\n".join(str(row[0]) for row in rows) + "\n")
        for name in ("regression", "binary"):
            path = directory / f"{name}.txt"
            model = lgb.Booster(model_file=str(path))
            expected = np.asarray(model.predict(rows, num_threads=1)).reshape(-1)
            output = subprocess.check_output([str(binary), "predict", "--input", str(input_path),
                       "--format", "csv", "--rows", str(len(rows)), "--features", "1", "--model", str(path)], text=True)
            actual = np.asarray([float(value) for value in output.splitlines()])
            assert actual.shape == expected.shape
            assert np.all(np.isfinite(actual))
            assert np.all(np.abs(actual - expected) <= 1e-12 + 1e-10 * np.abs(expected))
            cases.append(dict(name=name, model_sha256=sha(path), rows=len(rows),
                              rust_predictions=actual.tolist(), native_predictions=expected.tolist(),
                              max_absolute_difference=float(np.max(np.abs(actual - expected)))))
    save_json(ROOT / "tests/fixtures/rust-trained-model-roundtrip.json",
              dict(schema_version=1, upstream_commit=COMMIT, library=identity["library"],
                   rust_binary_sha256=sha(binary), kind="Rust-trained model interoperability, not a training-quality waiver", cases=cases))
    print("Native reference reloaded both Rust-trained models and matched all 16 predictions")


if __name__ == "__main__":
    main()
