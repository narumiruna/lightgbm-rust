"""Capture the pinned native Dataset/dense-prediction near-zero boundary split.

All scores, metrics, trees, and predictions come from the identity-checked native
CPU library. NumPy supplies input arrays and adjacent representable probes only.
"""
from pathlib import Path
import time

import numpy as np

from common import COMMIT, ROOT, SOURCE, check_resources, load_reference, save_json, sha

BASE = dict(objective='regression', metric='l2', verbosity=-1, num_threads=1,
            deterministic=True, force_col_wise=True, num_leaves=2,
            min_data_in_leaf=1, min_data_in_bin=1, feature_pre_filter=False,
            learning_rate=.1)
NAMES = {'negative_inclusive'}


def dense(model, rows):
    return dict(raw=np.asarray(model.predict(rows, raw_score=True, num_threads=1)).reshape(len(rows), 1).tolist(),
                leaves=np.asarray(model.predict(rows, pred_leaf=True, num_threads=1)).reshape(len(rows), -1).tolist())


def generate(fixtures: Path = ROOT / 'tests/fixtures'):
    fixtures = Path(fixtures)
    started = time.monotonic()
    lgb, record = load_reference()
    assert COMMIT == '5d84f3f222b41409a6983d2ac90513ad9702fa9e'
    assert record['library']['loaded_sha256'] == '9ed1d46d2d60bac2966fd03bd81e7d192025804f6a3e3ef24c3aa483d0ec7c1e'
    model_dir = fixtures / 'models/review_near_zero'
    model_dir.mkdir(parents=True, exist_ok=True)
    z = float(np.float32(1e-35))
    x = np.asarray([[-1.], [-z], [0.], [1.]], dtype=np.float64)
    y = np.asarray([0., 0., 1., 1.], dtype=np.float32)
    probes = np.asarray([[-1.], [np.nextafter(-z, -np.inf)], [-z],
                         [np.nextafter(-z, 0.)], [0.], [np.nextafter(z, 0.)],
                         [z], [np.nextafter(z, np.inf)], [1.]], dtype=np.float64)
    labels = np.asarray([0., 0., 0., 0., 1., 1., 1., 1., 1.], dtype=np.float32)
    data = lgb.Dataset(x, label=y, params=BASE)
    valid = lgb.Dataset(probes, label=labels, reference=data, params=BASE)
    buffers, history = {}, {}

    def capture(predictions, dataset):
        # Public feval obtains LGBM_BoosterGetPredict, not dense prediction.
        # The regression objective's ConvertOutput is the identity function.
        assert dataset is data or dataset is valid
        name = 'training' if dataset is data else 'validation'
        assert name not in buffers
        buffers[name] = predictions.copy().tolist()
        return []

    model = lgb.train(BASE, data, num_boost_round=1, valid_sets=[data, valid],
                      valid_names=['training', 'validation'], feval=capture,
                      callbacks=[lgb.record_evaluation(history)], keep_training_booster=True)
    assert set(buffers) == {'training', 'validation'}
    predictions = dict(training=dense(model, x), validation=dense(model, probes))
    structure = [tree['tree_structure'] for tree in model.dump_model()['tree_info']]
    filename = 'models/review_near_zero/negative_inclusive.txt'
    path = fixtures / filename
    assert path.resolve().parent == model_dir.resolve()
    model.save_model(str(path))
    saved = lgb.Booster(model_file=str(path))
    reloaded = dict(training=dense(saved, x), validation=dense(saved, probes))
    assert predictions == reloaded, 'saved native dense predictions changed'
    # Keep the native conflict as evidence. Do not replace Dataset raw binning
    # with dense prediction clipping to force these two native APIs to agree.
    assert buffers['training'][1] != predictions['training']['raw'][1][0]
    assert buffers['validation'][2] != predictions['validation']['raw'][2][0]
    cases = [dict(name='negative_inclusive', params=BASE, zero_threshold=z, rows=x.tolist(),
                  training=dict(rows=x.tolist(), labels=y.tolist(), num_boost_round=1,
                                scores=buffers['training'], metrics=history['training']),
                  validation=dict(rows=probes.tolist(), labels=labels.tolist(),
                                  scores=buffers['validation'], metrics=history['validation']),
                  model_file=filename, model_sha256=sha(path), native_structure=structure,
                  dense=predictions)]
    assert {case['name'] for case in cases} == NAMES
    sources = ('include/LightGBM/dataset.h', 'include/LightGBM/feature_group.h',
               'include/LightGBM/bin.h', 'include/LightGBM/meta.h', 'src/c_api.cpp',
               'src/boosting/gbdt.cpp', 'src/application/predictor.hpp',
               'src/objective/regression_objective.hpp', 'python-package/lightgbm/basic.py',
               'python-package/lightgbm/engine.py')
    save_json(fixtures / 'review_near_zero.json', dict(
        schema_version=1, upstream_commit=COMMIT, source=record['source'],
        library=record['library'], compiler=record['compiler'],
        compile_commands_sha256=record['compile_commands_sha256'],
        producer_sha256=sha(ROOT / 'tools/oracle/generate_review_near_zero.py'),
        source_sha256={name: sha(SOURCE / name) for name in sources},
        execution='Pinned native CPU serial; public feval captures GetPredict buffers; regression output is identity; keep_training_booster=True; NumPy input/output interchange and adjacent probes only',
        cases=cases))
    for name in ('training', 'validation'):
        scores = np.asarray(buffers[name])
        predicted = np.asarray(predictions[name]['raw']).reshape(-1)
        print(f'{name}: buffer/dense max_abs={np.max(np.abs(scores - predicted))!r}')
    print('live/saved dense max_abs=0.0; generated 1 actual native near-zero case')
    check_resources(started)


if __name__ == '__main__':
    generate()
