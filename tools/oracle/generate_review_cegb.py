"""Capture the pinned native CEGB pending-leaf refund behavior."""
from pathlib import Path
import time

import numpy as np

from common import COMMIT, ROOT, check_resources, load_reference, save_json, sha

BASE = dict(verbosity=-1, objective='regression', num_threads=1, force_col_wise=True,
            deterministic=True, seed=0, data_random_seed=0, feature_fraction_seed=0,
            min_data_in_bin=1, min_data_in_leaf=2, feature_pre_filter=False,
            num_leaves=4, max_bin=7, learning_rate=.2, cegb_tradeoff=1.)
COUPLED = [0., 5., 0.]
LAZY = [0., .2, 0.]


def specifications():
    mixed = dict(cegb_penalty_split=.1, cegb_penalty_feature_coupled=COUPLED,
                 cegb_penalty_feature_lazy=LAZY)
    return [
        ('coupled_pending', dict(cegb_penalty_feature_coupled=COUPLED)),
        ('coupled_split_pending', dict(cegb_penalty_split=.1, cegb_penalty_feature_coupled=COUPLED)),
        ('coupled_lazy_pending', dict(cegb_penalty_feature_coupled=COUPLED, cegb_penalty_feature_lazy=LAZY)),
        ('mixed_pending', mixed),
        ('mixed_rounds', dict(mixed, num_iterations=3)),
        ('mixed_contribution', dict(mixed, feature_contri=[1., .9, 1.])),
        ('mixed_node_mask', dict(mixed, feature_fraction_bynode=.75)),
        ('mixed_advanced', dict(mixed, monotone_constraints=[0, 1, 0],
                                monotone_constraints_method='advanced', monotone_penalty=.5)),
        ('zero_coupled', dict(mixed, cegb_penalty_feature_coupled=[0., 0., 0.])),
        ('zero_tradeoff', dict(mixed, cegb_tradeoff=0.)),
        ('zero_costs', dict(cegb_penalty_split=0., cegb_penalty_feature_coupled=[0., 0., 0.],
                            cegb_penalty_feature_lazy=[0., 0., 0.])),
        ('no_costs', {}),
    ]


NAMES = {name for name, _ in specifications()}


def generate(fixtures=ROOT / 'tests/fixtures'):
    fixtures = Path(fixtures)
    lgb, record = load_reference()
    assert COMMIT == '5d84f3f222b41409a6983d2ac90513ad9702fa9e'
    assert record['library']['loaded_sha256'] == '9ed1d46d2d60bac2966fd03bd81e7d192025804f6a3e3ef24c3aa483d0ec7c1e'
    model_dir = fixtures / 'models/review_cegb'
    model_dir.mkdir(parents=True, exist_ok=True)
    x = np.asarray([[g, b, c] for g in range(2) for b in range(2)
                    for c in range(2) for _ in range(2)], dtype=np.float64)
    y = np.asarray([20 * g + (4 if g == 0 else 2) * b + 2.5 * c * g
                    for g, b, c in x], dtype=np.float32)
    cases = []
    for name, extra in specifications():
        started = time.monotonic()
        assert name in NAMES and name.replace('_', '').isalnum()
        params = {**BASE, **extra}
        rounds = params.get('num_iterations', 1)
        data = lgb.Dataset(x, label=y, params=params)
        # Keep the live native model: engine.train otherwise serializes and
        # reloads it, which rounds diagnostic gains before dump_model().
        model = lgb.train(params, data, num_boost_round=rounds, keep_training_booster=True)
        filename = f'models/review_cegb/{name}.txt'
        path = fixtures / filename
        assert path.resolve().parent == model_dir.resolve()
        model.save_model(str(path))
        # Predictions come from the saved native model, not a Python estimator.
        saved = lgb.Booster(model_file=str(path))
        structure = [t['tree_structure'] for t in model.dump_model()['tree_info']]
        # Text output rounds split gains to six significant digits. Retain the
        # original native float gains as well as the unchanged native model.
        gains = []
        for root in structure:
            by_index = {}
            def visit(node):
                if 'split_index' in node:
                    by_index[node['split_index']] = node['split_gain']
                    visit(node['left_child'])
                    visit(node['right_child'])
            visit(root)
            gains.append([by_index[i] for i in range(len(by_index))])
        cases.append(dict(name=name, params=params, model_file=filename,
                          model_sha256=sha(path), native_split_gains=gains,
                          training=dict(rows=x.tolist(), labels=y.tolist(), num_boost_round=rounds,
                                        expected_structure=structure),
                          rows=x.tolist(),
                          raw=np.asarray(saved.predict(x, raw_score=True, num_threads=1)).reshape(len(x), 1).tolist(),
                          leaves=np.asarray(saved.predict(x, pred_leaf=True, num_threads=1)).reshape(len(x), -1).tolist()))
        check_resources(started)
    assert {c['name'] for c in cases} == NAMES
    save_json(fixtures / 'review_cegb.json', dict(
        schema_version=1, upstream_commit=COMMIT, source=record['source'],
        library=record['library'], compiler=record['compiler'],
        compile_commands_sha256=record['compile_commands_sha256'],
        producer_sha256=sha(ROOT / 'tools/oracle/generate_review_cegb.py'), cases=cases))
    print(f'Generated {len(cases)} actual native CEGB pending-leaf cases')


if __name__ == '__main__':
    generate()
