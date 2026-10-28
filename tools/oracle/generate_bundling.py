"""Regenerate native grouping and tiny bundled/unbundled training evidence."""
import json
import subprocess
import time
import numpy as np
from common import ROOT, OUT, load_reference, save_json, sha, check_resources
from native import compile_helper


def generate():
    started = time.monotonic()
    lgb, record = load_reference()
    assert record['library']['loaded_sha256'].startswith('9ed1d46d')
    exe, provenance = compile_helper('bundling_driver')
    base = dict(objective='regression', verbosity=-1, num_threads=1, force_col_wise=True,
                min_data_in_leaf=1, min_data_in_bin=1, max_bin=15, num_leaves=4,
                learning_rate=0.2, feature_pre_filter=True, seed=17)
    n, p = 40, 6
    exclusive = [[float(i % p == j) for j in range(p)] for i in range(n)]
    sparse = [[float(i == j) for j in range(p)] for i in range(n)]
    conflict = [row[:] for row in sparse]
    conflict[0][1] = 1.
    missing = [row[:] for row in exclusive]
    missing[0][0] = None
    frequent = [[3. if i % 10 else 0., float(i % 10 == 1), float(i % 10 == 2)] for i in range(n)]
    filtered = [[0., *row, 1.] for row in exclusive]
    categorical = [[2. if i % 4 else -1., float(i % 4 == 0), float(i % 4 == 1)] for i in range(n)]
    reordered = [[float(i % 2), float(i % 3), float(i % 5 == 0),
                  float(i % 5 == 1), float(i % 7 == 0), float(i % 7 == 1)] for i in range(64)]
    categorical_nan = [[None if i % 11 == 0 else row[0], *row[1:]] for i, row in enumerate(categorical)]
    inputs = []
    for name, rows, extra in [('exclusive', exclusive, {}), ('sparse', sparse, {}),
                              ('conflict', conflict, {}), ('missing', missing, {}),
                              ('zero_missing', missing, dict(zero_as_missing=True)),
                              ('frequent_nonzero', frequent, {}), ('filtered', filtered, {}),
                              ('categorical', categorical, dict(categorical_feature='0')),
                              ('categorical_nan', categorical_nan, dict(categorical_feature='0')),
                              ('reordered', reordered, {}),
                              ('extra_trees', reordered, dict(extra_trees=True, extra_seed=6))]:
        for enabled in [True, False]:
            params = dict(base, **extra, enable_bundle=enabled)
            text = ' '.join(f'{k}={str(v).lower() if isinstance(v, bool) else v}' for k, v in params.items())
            inputs.append(dict(name=f'{name}_{str(enabled).lower()}', rows=rows, parameters=text, params=params))
    run = subprocess.run([str(exe)], input=json.dumps(inputs, allow_nan=False), text=True,
                         capture_output=True, check=True, timeout=300)
    cases = json.loads(run.stdout)
    model_dir = ROOT / 'tests/fixtures/bundling_models'
    model_dir.mkdir(exist_ok=True)
    for case in cases:
        x = np.array([[np.nan if v is None else v for v in row] for row in case['rows']])
        y = np.array([float((i * 7) % 11) for i in range(len(x))], dtype=np.float32)
        case['labels'] = y.tolist()
        data = lgb.Dataset(x, label=y, params=case['params'],
                           categorical_feature=[0] if 'categorical_feature' in case['params'] else 'auto')
        booster = lgb.train(case['params'], data, num_boost_round=2)
        model = model_dir / f"{case['name']}.txt"
        model.write_text(booster.model_to_string())
        case['model'] = str(model.relative_to(ROOT / 'tests/fixtures'))
        case['model_sha256'] = sha(model)
        case['raw_predictions'] = booster.predict(x, raw_score=True).tolist()
    result = dict(schema_version=1, upstream_commit=record['source']['commit'],
                  loaded_library=record['library'], native_helper=provenance,
                  cases=cases, resources=check_resources(started, children=True))
    save_json(ROOT / 'tests/fixtures/bundling.json', result)
    return result


if __name__ == '__main__':
    generate()
