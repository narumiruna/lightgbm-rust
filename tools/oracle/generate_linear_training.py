"""Additional native linear-leaf training; preserve the original four cases."""
import time
import numpy as np

from common import COMMIT, ROOT, check_resources, load_reference, save_json
from generate_predictions import BASE, serial_rows

NAMES = {'linear_nan_training', 'linear_weighted_training', 'linear_nan_weighted_training'}


def generate(fixtures=ROOT / 'tests/fixtures'):
    lgb, _ = load_reference()
    cases = []
    for missing, weighted in [(True, False), (False, True), (True, True)]:
        started = time.monotonic()
        x = np.array([[float(i - 12), float((i * 7) % 11 - 5)] for i in range(24)])
        labels = np.asarray([1.2*a + 0.4*b + (3. if a > 0 else -2.) for a, b in x], dtype=np.float32)
        if missing:
            x[::5, 0] = np.nan
            x[::7, 1] = np.nan
        weights = np.asarray([1., 0.5, 2., 0., 3., 1.5] * 4, dtype=np.float32) if weighted else None
        name = 'linear_' + ('nan_' if missing else '') + ('weighted_' if weighted else '') + 'training'
        params = dict(BASE, objective='regression', linear_tree=True, linear_lambda=0.5, min_data_in_leaf=4)
        data = lgb.Dataset(x, label=labels, weight=weights, params=params)
        booster = lgb.train(params, data, num_boost_round=4)
        model_file = f'models/{name}.txt'
        booster.save_model(fixtures / model_file)
        model = lgb.Booster(model_file=str(fixtures / model_file))
        probes = np.vstack([x, [[np.nan, 0.], [0., np.nan], [np.nan, np.nan], [-20., -40.], [20., 40.]]])
        def pred(**kwargs):
            return np.asarray(model.predict(probes, num_threads=1, **kwargs)).reshape(len(probes), -1).tolist()
        training = dict(rows=serial_rows(x), labels=labels.tolist(), num_boost_round=4,
                        expected_structure=[t['tree_structure'] for t in model.dump_model()['tree_info']])
        if weighted:
            training['weights'] = weights.tolist()
        cases.append(dict(name=name, model_file=model_file, rows=serial_rows(probes), params=params,
                          raw=pred(raw_score=True), transformed=pred(),
                          leaves=np.asarray(pred(pred_leaf=True), dtype=np.int64).tolist(), training=training))
        check_resources(started)
    assert {c['name'] for c in cases} == NAMES
    save_json(fixtures / 'prediction_linear_training.json', dict(schema_version=1, upstream_commit=COMMIT, cases=cases))
    print('Generated three native NaN/weighted linear training cases')


if __name__ == '__main__':
    generate()
