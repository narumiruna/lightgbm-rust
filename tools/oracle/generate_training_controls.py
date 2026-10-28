"""Native tiny training for sampling, boosting updates, and split controls."""
import copy
import time

import numpy as np

from common import COMMIT, OUT, ROOT, check_resources, load_reference, save_json
from generate_predictions import BASE, serial_rows


def specifications():
    boosting = [
        ('bagging_rows', dict(bagging_fraction=0.7, bagging_freq=2)),
        ('bagging_balanced', dict(objective='binary', pos_bagging_fraction=0.5, neg_bagging_fraction=0.8, bagging_freq=1)),
        ('bagging_queries', dict(objective='lambdarank', bagging_fraction=0.7, bagging_freq=1, bagging_by_query=True)),
        ('feature_tree', dict(feature_fraction=0.5)),
        ('feature_node', dict(feature_fraction_bynode=0.5)),
        ('feature_tree_node', dict(feature_fraction=0.75, feature_fraction_bynode=0.5)),
        ('goss_warmup', dict(data_sample_strategy='goss', top_rate=0.25, other_rate=0.25)),
        ('goss_multiclass', dict(objective='multiclass', num_class=3, data_sample_strategy='goss', top_rate=0.25, other_rate=0.25)),
        ('dart_normalized', dict(boosting='dart', drop_rate=0.5, skip_drop=0., max_drop=3)),
        ('dart_uniform', dict(boosting='dart', drop_rate=0.5, skip_drop=0., max_drop=3, uniform_drop=True)),
        ('dart_xgboost', dict(boosting='dart', drop_rate=0.5, skip_drop=0., max_drop=3, xgboost_dart_mode=True)),
        ('rf_averaging', dict(boosting='rf', bagging_fraction=0.8, bagging_freq=1, feature_fraction=0.75)),
        ('rf_binary', dict(objective='binary', boosting='rf', bagging_fraction=0.8, bagging_freq=1, feature_fraction=0.75)),
    ]
    constraints = [
        ('controls_baseline', {}),
        *[(f'monotone_{method}', dict(monotone_constraints=[1, -1, 0, 0], monotone_constraints_method=method))
          for method in ('basic', 'intermediate', 'advanced')],
        ('monotone_penalty', dict(monotone_constraints=[1, -1, 0, 0], monotone_constraints_method='advanced', monotone_penalty=1.5)),
        ('interaction', dict(interaction_constraints=[[0, 1], [2, 3]])),
        ('extra_trees', dict(extra_trees=True, extra_seed=42)),
        ('forced_splits', dict(forcedsplits_filename=str(OUT / 'controls-forced-splits.json'))),
        ('path_smooth', dict(path_smooth=2.)),
        ('feature_penalties', dict(feature_contri=[0.1, 1., 0., 2.])),
        ('cegb_split', dict(cegb_tradeoff=1., cegb_penalty_split=1.)),
        ('cegb_coupled', dict(cegb_tradeoff=1., cegb_penalty_feature_coupled=[400., 0., 25., 20.])),
        ('cegb_lazy', dict(cegb_tradeoff=1., cegb_penalty_feature_lazy=[12.5, 0., 2., 1.])),
        ('controls_combined', dict(extra_trees=True, extra_seed=42, path_smooth=2., feature_contri=[0.5, 1., 1., 0.5], interaction_constraints=[[0, 1], [2, 3]])),
        ('quantized_deterministic', dict(use_quantized_grad=True, num_grad_quant_bins=4, stochastic_rounding=False, quant_train_renew_leaf=False)),
        ('quantized_stochastic', dict(use_quantized_grad=True, num_grad_quant_bins=4, stochastic_rounding=True, quant_train_renew_leaf=False, seed=42)),
        ('quantized_renew', dict(use_quantized_grad=True, num_grad_quant_bins=4, stochastic_rounding=True, quant_train_renew_leaf=True, seed=42)),
        ('quantized_binary', dict(objective='binary', use_quantized_grad=True, num_grad_quant_bins=8, stochastic_rounding=True, quant_train_renew_leaf=True, seed=42)),
    ]
    return dict(prediction_boosting=boosting, prediction_constraints=constraints)


def generate(fixtures=ROOT / 'tests/fixtures'):
    lgb, _ = load_reference()
    (fixtures / 'models').mkdir(parents=True, exist_ok=True)
    save_json(OUT / 'controls-forced-splits.json', dict(feature=1, threshold=0., left=dict(feature=0, threshold=0.)))
    x = np.asarray([[float(i - 16), float((i * 7) % 13 - 6), float((i * 3) % 7 - 3), float((i * 5) % 11 - 5)] for i in range(32)])
    probes = np.vstack([x, [[np.nan, 0., 0., 0.], [0., np.nan, 0., 0.], [0., 0., np.nan, np.nan]]])
    weights = np.asarray([1., 0.5, 2., 0., 1., 1.5, 0.25, 3.] * 4, dtype=np.float32)
    for family, specs in specifications().items():
        cases = []
        for name, extra in specs:
            started = time.monotonic()
            params = dict(BASE)
            params.update(objective='regression', num_leaves=5, min_data_in_leaf=2, max_bin=15)
            params.update(extra)
            labels = np.asarray([0.3*a - 0.5*b + 0.8*(c > 0) + 0.15*d + (2. if a > 3 else -1.) for a, b, c, d in x], dtype=np.float32)
            groups = positions = None
            if params['objective'] == 'binary':
                labels = (labels > 0).astype(np.float32)
            elif params['objective'] == 'multiclass':
                labels = np.asarray([i % 3 for i in range(len(x))], dtype=np.float32)
            elif params['objective'] == 'lambdarank':
                labels = np.asarray([3., 1., 0., 2.] * 8, dtype=np.float32)
                groups = [4] * 8
                positions = np.asarray([7, 3, 7, 3] * 8, dtype=np.int32)
            rounds = 9 if family == 'prediction_boosting' else 4
            data = lgb.Dataset(x, label=labels, weight=weights, group=groups, position=positions, params=params)
            steps = []
            def record_step(env):
                dump = env.model.dump_model()
                steps.append(dict(iteration=env.iteration, raw=np.asarray(env.model.predict(probes, raw_score=True, num_threads=1)).reshape(len(probes), -1).tolist(),
                                  tree_shrinkage=[tree['shrinkage'] for tree in dump['tree_info']]))
            record_step.order = 50
            record_step.before_iteration = False
            booster = lgb.train(params, data, num_boost_round=rounds, callbacks=[record_step])
            model_file = f'models/{name}.txt'
            booster.save_model(fixtures / model_file)
            model = lgb.Booster(model_file=str(fixtures / model_file))
            def pred(**kwargs):
                return np.asarray(model.predict(probes, num_threads=1, **kwargs)).reshape(len(probes), -1).tolist()
            training = dict(rows=serial_rows(x), labels=labels.tolist(), weights=weights.tolist(), num_boost_round=rounds,
                            expected_structure=[t['tree_structure'] for t in model.dump_model()['tree_info']], steps=copy.deepcopy(steps))
            if groups is not None:
                training.update(groups=groups, positions=positions.tolist())
            cases.append(dict(name=name, model_file=model_file, rows=serial_rows(probes), params=params,
                              raw=pred(raw_score=True), transformed=pred(), leaves=np.asarray(pred(pred_leaf=True), dtype=np.int64).tolist(), training=training))
            check_resources(started)
        save_json(fixtures / f'{family}.json', dict(schema_version=1, upstream_commit=COMMIT, cases=cases))
        print(f'Generated {len(cases)} native {family} training cases with per-round scores/shrinkage')


if __name__ == '__main__':
    generate()
