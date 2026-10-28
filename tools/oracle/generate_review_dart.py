"""R3: original native successful prefixes and safe DART no-split evidence.

The seed-4 empty second bag is NOT passed to native TrainOneIter. Seed 11
uses a nonempty, constant-label second bag to observe the same early return.
No replacement learner, normalization formula, or failed-round parity is used.
"""
import hashlib
from pathlib import Path
import time

import numpy as np

from common import COMMIT, ROOT, SOURCE, check_resources, load_reference, save_json, sha
from generate_predictions import BASE
from workflow_common import metrics

FIXTURES = ROOT / 'tests/fixtures'
MODES = ('uniform', 'weighted', 'xgboost', 'no_drop')
NAMES = {f'review_dart_{kind}_{mode}'
         for kind in ('empty', 'safe') for mode in MODES}
NAMES |= {f'review_dart_continue_{mode}' for mode in ('uniform', 'weighted')}
NAMES |= {f'review_dart_multiclass_{mode}' for mode in ('uniform', 'weighted')}
NAMES.add('review_dart_initial_constant')


def specifications(fixtures=FIXTURES):
    # fixtures is accepted for the common producer interface; inputs are fixed.
    Path(fixtures)
    cases = []
    variants = [(kind, mode) for kind in ('empty', 'safe') for mode in MODES]
    variants += [('continue', mode) for mode in ('uniform', 'weighted')]
    variants += [('multiclass', mode) for mode in ('uniform', 'weighted')]
    variants += [('initial', 'constant')]
    for kind, mode in variants:
        name = f'review_dart_{kind}_{mode}'
        params = dict(BASE, boosting='dart', objective='regression', metric='l2',
                      bagging_fraction=0.5, bagging_freq=1,
                      bagging_seed=4 if kind in ('empty', 'continue') else 11,
                      uniform_drop=mode != 'weighted', xgboost_dart_mode=mode == 'xgboost',
                      drop_rate=1.0, skip_drop=1.0 if mode == 'no_drop' else 0.0)
        if kind == 'multiclass':
            params.update(objective='multiclass', num_class=3, metric='multi_logloss')
        cases.append(dict(name=name, params=params, rows=[[0.0], [1.0], [2.0], [3.0]],
                          labels=[2, 2, 2, 2] if kind == 'initial' else [0, 0, 1, 1],
                          initial_model=kind == 'continue',
                          native_attempts=1 if kind in ('empty', 'continue', 'initial') else 2,
                          rust_attempts=6,
                          bag_rows=[[0, 3], []] if params['bagging_seed'] == 4 else [[0, 3], [2, 3]],
                          native_empty_bag_executed=False))
    assert {c['name'] for c in cases} == NAMES
    return dict(schema_version=1, upstream_commit=COMMIT, cases=cases)


def model_path(fixtures, name):
    assert name in NAMES or name.removesuffix('_initial') in NAMES or name.removesuffix('_last_valid') in NAMES
    relative = Path('models/review_dart') / f'{name}.txt'
    root = Path(fixtures).resolve()
    destination = root / relative
    assert destination.resolve().is_relative_to(root), 'unsafe model path'
    destination.parent.mkdir(parents=True, exist_ok=True)
    return relative.as_posix(), destination


def snapshot(model, rows):
    text = model.model_to_string(num_iteration=-1)
    return dict(current_iteration=model.current_iteration(), num_trees=model.num_trees(),
                shrinkage=[float(line.split('=', 1)[1]) for line in text.splitlines()
                           if line.startswith('shrinkage=')],
                raw=np.asarray(model.predict(rows, raw_score=True, num_iteration=-1,
                                             num_threads=1)).reshape(len(rows), -1).tolist(),
                # Read only after update: GetTrainingScore has already dropped
                # once in this iteration. Never use it as a before-round probe.
                training_predictions=model._Booster__inner_predict(data_idx=0).tolist(),
                evaluation=metrics(model.eval_train() + model.eval_valid()))


def save_model(model, fixtures, name):
    relative, destination = model_path(fixtures, name)
    model.save_model(destination, num_iteration=-1)
    return dict(model_file=relative, sha256=sha(destination))


def generate(fixtures=FIXTURES):
    fixtures = Path(fixtures)
    lgb, record = load_reference()
    payload = specifications(fixtures)
    for case in payload['cases']:
        started = time.monotonic()
        x = np.asarray(case['rows'], dtype=np.float64)
        y = np.asarray(case['labels'], dtype=np.float32)
        params = case['params']
        prefix = None
        if case['initial_model']:
            data = lgb.Dataset(x, label=y, params=params)
            prefix = lgb.Booster(params=params, train_set=data)
            assert prefix.update() is False
            case['initial_model'] = save_model(prefix, fixtures, case['name'] + '_initial')
        else:
            case['initial_model'] = None
        data = lgb.Dataset(x, label=y, params=params, free_raw_data=False)
        valid = lgb.Dataset(x[::-1].copy(), label=y[::-1].copy(), reference=data)
        if prefix is not None:
            # Match engine.train stored-model continuation without RNG checkpoint.
            predictor = lgb.basic._InnerPredictor.from_booster(prefix, pred_parameter={})
            data._set_predictor(predictor)
            valid._set_predictor(predictor)
        model = lgb.Booster(params=params, train_set=data)
        model.add_valid(valid, 'valid')
        case['native_steps'] = []
        for attempt in range(case['native_attempts']):
            # No empty native learner input. These bags are checked in the
            # specifications and independently in the Rust RNG regression.
            assert case['bag_rows'][attempt]
            finished = model.update()
            step = dict(finished=finished, **snapshot(model, x))
            case['native_steps'].append(step)
            if attempt == 0:
                case['last_valid_model'] = save_model(model, fixtures, case['name'] + '_last_valid')
            if finished:
                break
        if any(value < 0 for value in case['native_steps'][-1]['shrinkage']):
            text = model.model_to_string(num_iteration=-1)
            case['native_final_model'] = None
            case['native_failed_model'] = dict(text=text, sha256=hashlib.sha256(text.encode()).hexdigest())
        else:
            case['native_final_model'] = save_model(model, fixtures, case['name'])
            case['native_failed_model'] = None
        if case['native_attempts'] == 2:
            assert case['native_steps'][0]['finished'] is False
            if params['skip_drop'] == 0:
                assert case['native_steps'][1]['finished'] is True
                assert any(v < 0 for v in case['native_steps'][1]['shrinkage'])
            else:
                assert case['native_steps'][1]['finished'] is False
        check_resources(started)
    payload['provenance'] = dict(source=record['source'], compiler=record['compiler'],
                                 producer_sha256=sha(Path(__file__)),
                                 python_sources={'basic.py': sha(SOURCE / 'python-package/lightgbm/basic.py')},
                                 compile_commands_sha256=record['compile_commands_sha256'],
                                 library=record['library'],
                                 native_sources={p: sha(SOURCE / p) for p in (
                                     'src/boosting/dart.hpp', 'src/boosting/gbdt.cpp',
                                     'src/boosting/bagging.hpp', 'include/LightGBM/utils/random.h')})
    payload['checked_departure'] = (
        'Native DART::TrainOneIter returns on GBDT no-split before Normalize. '
        'Safe nonempty-bag evidence retains negative shrinkage and changed training scores; '
        'validation scores remain at the completed prefix. Rust restores dropped trees and '
        'training scores, leaves validation unchanged, and does not call DartState::finish. '
        'Rust expected failed-round state is the actual native last_valid_model, NOT the '
        'native failed-round model. Dropout and sampler RNG draws remain consumed. '
        'Seed-4 second empty native learner input is excluded; only its successful prefix runs.')
    save_json(fixtures / 'review_dart.json', payload)
    print(f'Generated {len(payload["cases"])} R3 native cases')


if __name__ == '__main__':
    generate()
