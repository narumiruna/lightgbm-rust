"""Capture native R1/R5 live predictions, refit, and six-digit sigmoid metadata.

NumPy supplies input/output arrays only. All objectives, refits, and predictions
run in the identity-checked pinned native library. No helper executable is used.
"""
from pathlib import Path
import time

import numpy as np

from common import COMMIT, ROOT, SOURCE, check_resources, load_reference, save_json, sha
from generate_predictions import BASE
from workflow_common import predict, save_model, snapshot


def specifications():
    return [(f'{objective}_{style}', dict(objective=objective, sigmoid=sigmoid,
                                        num_class=3 if objective == 'multiclassova' else 1))
            for objective in ('binary', 'multiclassova')
            for style, sigmoid in (('decimal', 1.23456789), ('scientific', 0.0000123456789))]


NAMES = {name for name, _ in specifications()}


def generate(fixtures: Path = ROOT / 'tests/fixtures'):
    lgb, record = load_reference()
    assert COMMIT == '5d84f3f222b41409a6983d2ac90513ad9702fa9e'
    assert record['library']['loaded_sha256'] == '9ed1d46d2d60bac2966fd03bd81e7d192025804f6a3e3ef24c3aa483d0ec7c1e'
    (fixtures / 'models/review_refit').mkdir(parents=True, exist_ok=True)
    rows = [[float(i - 12), float((i * 5) % 11 - 5)] for i in range(24)]
    new_rows = [[a + (1.0 if i % 3 == 0 else 0.0), b] for i, (a, b) in enumerate(rows)]
    weights = [1.0, 2.0, 0.5, 1.0] * 6
    x, new_x = np.asarray(rows, dtype=np.float64), np.asarray(new_rows, dtype=np.float64)
    cases = []
    for name, extra in specifications():
        started = time.monotonic()
        params = dict(BASE, min_sum_hessian_in_leaf=0.0, **extra)
        labels = [i % 3 for i in range(24)] if extra['num_class'] == 3 else [int(a > 0) for a, _ in rows]
        new_labels = labels[::-1]
        data = lgb.Dataset(x, label=np.asarray(labels, dtype=np.float32),
                           weight=np.asarray(weights, dtype=np.float32), params=params)
        fresh = lgb.train(params, data, num_boost_round=4, keep_training_booster=True)
        live_initial = snapshot(fresh, new_x)
        initial = save_model(fresh, fixtures, f'review_refit/{name}_initial', new_x)
        assert live_initial['raw'] == initial['raw']
        assert live_initial['transformed'] != initial['transformed']
        loaded = lgb.Booster(model_file=str(fixtures / initial['model_file']))
        native_refit = {}
        for mode, model in (('fresh', fresh), ('reloaded', loaded)):
            result = model.refit(new_x, np.asarray(new_labels, dtype=np.float32), decay_rate=0.5,
                                 weight=np.asarray(weights, dtype=np.float32), dataset_params=params)
            native_refit[mode] = snapshot(result, new_x)
        assert native_refit['fresh'] == native_refit['reloaded']
        final = save_model(result, fixtures, f'review_refit/{name}_final', new_x)
        # A second actual native refit proves that derivative work must not use
        # the rounded metadata value in place of the caller's configuration.
        objective = next(line.split('=', 1)[1] for line in result.model_to_string().splitlines()
                         if line.startswith('objective='))
        serialized_sigmoid = float(next(t.split(':', 1)[1] for t in objective.split()
                                       if t.startswith('sigmoid:')))
        rounded = loaded.refit(new_x, np.asarray(new_labels, dtype=np.float32), decay_rate=0.5,
                               weight=np.asarray(weights, dtype=np.float32),
                               dataset_params=dict(params, sigmoid=serialized_sigmoid))
        rounded_refit = snapshot(rounded, new_x)
        assert rounded_refit['raw'] != native_refit['fresh']['raw']
        cases.append(dict(name=name, params=params,
                          training=dict(rows=rows, labels=labels, weights=weights, num_boost_round=4),
                          rows=new_rows, labels=new_labels, weights=weights, decay_rate=0.5,
                          objective=objective, serialized_sigmoid=serialized_sigmoid,
                          distinct_sigmoid=extra['sigmoid'] * 1.00002,
                          initial_model=initial, live_initial=live_initial,
                          leaf_pred=predict(loaded, new_x, pred_leaf=True),
                          final_model=final, native_refit=native_refit, rounded_refit=rounded_refit))
        check_resources(started)
    assert {c['name'] for c in cases} == NAMES
    sources = ('src/objective/binary_objective.hpp', 'src/objective/multiclass_objective.hpp',
               'src/boosting/gbdt.cpp', 'src/treelearner/serial_tree_learner.cpp',
               'python-package/lightgbm/basic.py')
    save_json(fixtures / 'review_refit.json', dict(
        schema_version=1, upstream_commit=COMMIT, source=record['source'],
        library=record['library'], compiler=record['compiler'],
        compile_commands_sha256=record['compile_commands_sha256'],
        producer_sha256=sha(ROOT / 'tools/oracle/generate_review_refit.py'),
        source_sha256={path: sha(SOURCE / path) for path in sources},
        execution='Pinned native CPU serial; NumPy input/output interchange only; no helper executable',
        cases=cases))
    print(f'Generated {len(cases)} actual native review refit cases')


if __name__ == '__main__':
    generate()
