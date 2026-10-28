"""Tiny native training for extended objectives/categories and native CSR inference."""
import time
import numpy as np
from scipy.sparse import csr_matrix

from common import COMMIT, ROOT, check_resources, load_reference, save_json
from generate_predictions import BASE, serial_rows
from generate_extended import OBJECTIVES


def generate(fixtures=ROOT / 'tests/fixtures'):
    lgb, _ = load_reference()
    (fixtures / 'models').mkdir(parents=True, exist_ok=True)
    x = np.array([[i - 5., (i * 3) % 5] for i in range(12)], dtype=np.float64)
    probe = np.vstack([x, [[np.nan, 0.], [1e-36, 1.], [-1e-36, 0.], [0., np.nan]]])
    specs = []
    weights = np.asarray([1., 0.5, 2., 1., 3., 0.25] * 2, dtype=np.float32)
    for objective in OBJECTIVES:
        labels = np.asarray([-3., -1., -2., 0., 0.5, 1., 3., 2., 4., 3., 5., 7.], dtype=np.float32)
        extra = {}
        if objective in ('poisson', 'gamma', 'tweedie'):
            labels = labels - min(labels) + 0.25
        if objective in ('binary', 'multiclass', 'multiclassova'):
            labels = np.array([0., 0., 1., 1., 2., 2., 0., 1., 2., 0., 1., 2.], dtype=np.float32)
            if objective == 'binary':
                labels = (labels > 0).astype(np.float32)
            else:
                extra['num_class'] = 3
        if objective in ('cross_entropy', 'cross_entropy_lambda'):
            labels = np.array([0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1., 1.], dtype=np.float32)
        groups = positions = None
        if objective in ('lambdarank', 'rank_xendcg'):
            labels = np.array([3., 1., 0., 2., 0., 1., 0., 3., 0., 0., 1., 2.], dtype=np.float32)
            groups = [4, 4, 4]
            if objective == 'lambdarank':
                positions = np.array([7, 3, 7, 3] * 3, dtype=np.int32)
        specs.append((objective + '_extended', objective, x, labels, probe, weights, groups, positions, extra))
        if objective in ('multiclass', 'multiclassova'):
            specs.append((objective + '_absent_extended', objective, x, (labels == 1.).astype(np.float32), probe, weights, None, None, extra))
    cat = np.array([[float(c)] for c in range(6) for _ in range(4)] + [[np.nan], [-1.]], dtype=np.float64)
    cat_labels = np.array([float([0, 5, 1, 6, 2, 7][c]) for c in range(6) for _ in range(4)] + [4., 4.], dtype=np.float32)
    cat_weights = np.array(([1., 2., 0.5, 1.5] * 6) + [2., 0.5], dtype=np.float32)
    cat_probe = np.vstack([cat, [[8.], [1.9], [-0.5], [-2.], [0.]]])
    for partition in (False, True):
        for weighted in (False, True):
            name = 'categorical_' + ('partition' if partition else 'onehot') + ('_weighted' if weighted else '')
            extra = dict(categorical_feature=[0], max_cat_to_onehot=2 if partition else 8,
                         min_data_per_group=1, max_cat_threshold=8, cat_smooth=0., cat_l2=0.)
            specs.append((name, 'regression', cat, cat_labels, cat_probe, cat_weights if weighted else None, None, None, extra))
    cases = []
    for name, objective, rows, labels, probes, weight, groups, positions, extra in specs:
        started = time.monotonic()
        params = dict(BASE, objective=objective, **extra)
        kw = {}
        if 'categorical_feature' in extra:
            kw['categorical_feature'] = extra['categorical_feature']
        ds = lgb.Dataset(rows, label=labels, weight=weight, group=groups, position=positions, params=params, **kw)
        booster = lgb.train(params, ds, num_boost_round=3)
        model_file = 'models/' + name + '.txt'
        booster.save_model(fixtures / model_file)
        model = lgb.Booster(model_file=str(fixtures / model_file))
        def pred(**kwargs):
            return np.asarray(model.predict(probes, num_threads=1, **kwargs)).reshape(len(probes), -1).tolist()
        structure = [t['tree_structure'] for t in model.dump_model()['tree_info']]
        if name.startswith('categorical_partition'):
            def partitions(node):
                if 'split_index' not in node:
                    return []
                return ([node['threshold']] if node['decision_type'] == '==' else []) + partitions(node['left_child']) + partitions(node['right_child'])
            assert any('||' in p for tree in structure for p in partitions(tree)), 'sorted partition not exercised'
        training = dict(rows=serial_rows(rows), labels=labels.tolist(), num_boost_round=3, expected_structure=structure)
        for key, value in [('weights', weight), ('groups', groups), ('positions', positions)]:
            if value is not None:
                training[key] = np.asarray(value).tolist()
        cases.append(dict(name=name, model_file=model_file, rows=serial_rows(probes), raw=pred(raw_score=True),
                          transformed=pred(), leaves=np.asarray(pred(pred_leaf=True), dtype=np.int64).tolist(),
                          params=params, training=training))
        check_resources(started)
    save_json(fixtures / 'prediction_extended.json', dict(schema_version=1, upstream_commit=COMMIT, cases=cases))
    sparse = []
    import json
    base = json.loads((ROOT / 'tests/fixtures/prediction.json').read_text())
    for c in base['cases']:
        started = time.monotonic()
        rows = np.array([[np.nan if v is None else v for v in row] for row in c['rows']], dtype=np.float64)
        matrix = csr_matrix(rows)
        model = lgb.Booster(model_file=str(ROOT / 'tests/fixtures' / c['model_file']))
        def pred(**kwargs):
            return np.asarray(model.predict(matrix, num_threads=1, **kwargs)).reshape(len(rows), -1).tolist()
        sparse.append(dict(name=c['name'] + '_csr', model_file=c['model_file'], rows=c['rows'], params=c['params'],
                           sparse=dict(indptr=matrix.indptr.tolist(), indices=matrix.indices.tolist(),
                                       data=[None if np.isnan(v) else float(v) for v in matrix.data], shape=list(matrix.shape)),
                           raw=pred(raw_score=True), transformed=pred(),
                           leaves=np.asarray(pred(pred_leaf=True), dtype=np.int64).tolist()))
        check_resources(started)
    save_json(fixtures / 'prediction_sparse.json', dict(schema_version=1, upstream_commit=COMMIT, cases=sparse))
    print(f'Generated {len(cases)} native tiny training and {len(sparse)} native CSR cases')


if __name__ == '__main__':
    generate()
