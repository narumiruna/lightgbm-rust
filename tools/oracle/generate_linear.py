"""Generate native linear-leaf v4 model evidence, including singular inputs."""
import time
import numpy as np

from common import COMMIT, OUT, ROOT, check_resources, load_reference, save_json
from generate_predictions import BASE, serial_rows


def generate(fixtures=ROOT / 'tests/fixtures'):
    lgb, _ = load_reference()
    (fixtures / 'models').mkdir(parents=True, exist_ok=True)
    forced_path = OUT / 'linear-forced-splits.json'
    save_json(forced_path, dict(feature=0, threshold=0., left=dict(feature=1, threshold=-12.),
                               right=dict(feature=1, threshold=12.)))
    cases = []
    for singular in (False, True):
        for regularization in (0., 0.5):
            started = time.monotonic()
            x = np.array([[float(i - 12), float((i * 7) % 11 - 5)] for i in range(24)])
            if singular:
                x[:, 1] = 2. * x[:, 0]
            labels = np.asarray([1.2*a + 0.4*b + (3. if a > 0 else -2.) for a, b in x], dtype=np.float32)
            probes = np.vstack([x, [[np.nan, 0.], [0., np.nan], [np.nan, np.nan], [-20., -40.], [20., 40.]]])
            name = 'linear_' + ('rank_deficient' if singular else 'independent') + ('_lambda05' if regularization else '_lambda0')
            params = dict(BASE, objective='regression', linear_tree=True, linear_lambda=regularization,
                          min_data_in_leaf=4)
            if singular:
                params.update(forcedsplits_filename=str(forced_path), num_leaves=4)
            ds = lgb.Dataset(x, label=labels, params=params)
            booster = lgb.train(params, ds, num_boost_round=4)
            model_file = 'models/' + name + '.txt'
            booster.save_model(fixtures / model_file)
            model = lgb.Booster(model_file=str(fixtures / model_file))
            def pred(**kwargs):
                return np.asarray(model.predict(probes, num_threads=1, **kwargs)).reshape(len(probes), -1).tolist()
            structure = [t['tree_structure'] for t in model.dump_model()['tree_info']]
            if singular:
                def branch_features(node, features=frozenset()):
                    if 'left_child' not in node:
                        return [features]
                    features = features | {node['split_feature']}
                    return branch_features(node['left_child'], features) + branch_features(node['right_child'], features)
                # Native solving can prune a zero coefficient. Check the input path,
                # not the post-solve leaf_features array, to prove singularity.
                assert any(len(path) == 2 for tree in structure[1:] for path in branch_features(tree)), 'singular two-feature path not exercised'
            cases.append(dict(name=name, model_file=model_file, rows=serial_rows(probes), params=params,
                              raw=pred(raw_score=True), transformed=pred(),
                              leaves=np.asarray(pred(pred_leaf=True), dtype=np.int64).tolist(),
                              training=dict(rows=serial_rows(x), labels=labels.tolist(), num_boost_round=4,
                                            expected_structure=structure)))
            check_resources(started)
    save_json(fixtures / 'prediction_linear.json', dict(schema_version=1, upstream_commit=COMMIT, cases=cases))
    print('Generated four native linear-leaf models with first-tree, regularization, singularity, NaN fallback probes')


if __name__ == '__main__':
    generate()
