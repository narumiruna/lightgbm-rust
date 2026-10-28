"""Shared serialization for actual Python engine and native model operations."""
import importlib.metadata
import math

import numpy as np

from common import COMMIT, SOURCE, sha


def provenance(record):
    return dict(upstream_commit=COMMIT, loaded_library_sha256=record['library']['loaded_sha256'],
                python_sources={name: sha(SOURCE / 'python-package/lightgbm' / name)
                                for name in ('engine.py', 'callback.py', 'basic.py', 'compat.py')},
                dependencies={name: importlib.metadata.version(name) for name in ('numpy', 'scipy', 'scikit-learn')},
                execution='native CPU serial; original Python engine/callback; NumPy interchange and upstream fold/aggregation only',
                iteration_index='zero-based callback iteration; one-based best_iteration; 0 means no selected best in train, -1 in cv')


def metrics(values):
    if values is None:
        return None
    return [dict(dataset_name=v.dataset_name, metric_name=v.metric_name, metric_value=float(v.metric_value),
                 maximize=bool(v.maximize), metric_std_dev=None if v.metric_std_dev is None else float(v.metric_std_dev)) for v in values]


def predict(model, rows, **kwargs):
    values = np.asarray(model.predict(rows, num_threads=1, **kwargs))
    return values.reshape(len(rows), values.size // len(rows)).tolist()


def snapshot(model, rows):
    return dict(current_iteration=model.current_iteration(), num_trees=model.num_trees(),
                raw=predict(model, rows, raw_score=True, num_iteration=-1),
                transformed=predict(model, rows, num_iteration=-1),
                leaves=predict(model, rows, pred_leaf=True, num_iteration=-1))


def save_model(model, fixtures, name, rows, num_iteration=None):
    path = f'models/{name}.txt'
    model.save_model(fixtures / path, num_iteration=num_iteration)
    # Always report predictions from the exact saved bytes.
    from lightgbm import Booster
    loaded = Booster(model_file=str(fixtures / path))
    return dict(model_file=path, **snapshot(loaded, rows))


def exceptional(value):
    if isinstance(value, dict):
        return {k: exceptional(v) for k, v in value.items()}
    if isinstance(value, list):
        return [exceptional(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return 'NaN' if math.isnan(value) else 'Infinity' if value > 0 else '-Infinity'
    return value


def integer_data():
    rows = [[i - 12, (i * 5) % 11 - 5] for i in range(24)]
    labels = [2 * a - b + (3 if a > 0 else -2) for a, b in rows]
    return np.asarray(rows, dtype=np.float64), labels
