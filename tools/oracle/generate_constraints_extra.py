"""Regenerate the ten additional native learner-control cases without changing model bytes."""
import time

import numpy as np

from common import COMMIT, ROOT, check_resources, load_reference, save_json, sha

NAMES = {'deep_monotone_' + method for method in ('basic', 'intermediate', 'advanced', 'smooth')} | {
    'cat_smooth', 'cat_extra_onehot', 'cat_extra_sorted', 'cat_quantized', 'cat_quantized_renew', 'interaction_sampled'}
BASE = dict(verbosity=-1, objective='regression', num_threads=1, force_col_wise=True, deterministic=True,
            seed=0, data_random_seed=0, feature_fraction_seed=0, extra_seed=42, min_data_in_bin=1,
            min_data_in_leaf=2, feature_pre_filter=False, num_leaves=9, max_bin=15, learning_rate=.2)


def specifications():
    specs = [('deep_monotone_' + method, dict(monotone_constraints=[1, -1, 0], monotone_constraints_method=method))
             for method in ('basic', 'intermediate', 'advanced')]
    specs += [('deep_monotone_smooth', dict(monotone_constraints=[1, -1, 0], monotone_constraints_method='advanced', path_smooth=2.)),
              ('cat_smooth', dict(categorical_feature=[2], cat_smooth=0., cat_l2=1., min_data_per_group=1, path_smooth=3.)),
              ('cat_extra_onehot', dict(categorical_feature=[2], max_cat_to_onehot=8, extra_trees=True, cat_smooth=0., min_data_per_group=1)),
              ('cat_extra_sorted', dict(categorical_feature=[2], max_cat_to_onehot=2, extra_trees=True, cat_smooth=0., min_data_per_group=1)),
              ('cat_quantized', dict(categorical_feature=[2], cat_smooth=0., min_data_per_group=1, use_quantized_grad=True,
                                     num_grad_quant_bins=8, stochastic_rounding=True, seed=42)),
              ('cat_quantized_renew', dict(categorical_feature=[2], cat_smooth=0., min_data_per_group=1, use_quantized_grad=True,
                                           num_grad_quant_bins=8, stochastic_rounding=True, quant_train_renew_leaf=True, seed=42)),
              ('interaction_sampled', dict(interaction_constraints=[[0, 2], [1]], feature_fraction=.8, feature_fraction_bynode=.7, extra_trees=True))]
    return specs


def schema():
    number = dict(type='number')
    vector = dict(type='array', minItems=1, items=number)
    matrix = dict(type='array', minItems=1, items=vector)
    def obj(properties):
        return dict(type='object', properties=properties, required=list(properties), additionalProperties=False)
    digest = dict(type='string', pattern='^[a-f0-9]{64}$')
    case = obj(dict(name=dict(type='string', enum=sorted(NAMES)), params=dict(type='object', minProperties=1),
                    model_file=dict(type='string', pattern=r'^models/constraints_extra/[a-z_]+\.txt$'),
                    training=obj(dict(rows=matrix, labels=vector, weights=vector, num_boost_round=dict(const=4))),
                    rows=matrix, raw=matrix,
                    leaves=dict(type='array', minItems=1, items=dict(type='array', minItems=1, items=dict(type='integer', minimum=0)))))
    return dict({'$schema': 'https://json-schema.org/draft/2020-12/schema'}, **obj(dict(
        schema_version=dict(const=1), upstream_commit=dict(const=COMMIT),
        library=dict(type='object', required=['path', 'sha256', 'loaded_path', 'loaded_sha256', 'verification'],
                     properties={key: digest for key in ('sha256', 'loaded_sha256')}),
        compiler=dict(type='object', required=['path', 'sha256', 'version', 'selected_toolchain_path', 'selected_toolchain_sha256']),
        compile_commands_sha256=digest, producer_sha256=digest, cases=dict(type='array', minItems=10, maxItems=10, items=case))))


def generate(fixtures=ROOT / 'tests/fixtures'):
    lgb, record = load_reference()
    (fixtures / 'models/constraints_extra').mkdir(parents=True, exist_ok=True)
    x = np.asarray([[float(i), float(j), float((i * 3 + j * 7) % 5)] for i in range(10) for j in range(10)])
    y = np.asarray([.3 * i - .2 * j + np.sin(i + j) + (.8 if c < 2 else -.8) for i, j, c in x], dtype=np.float32)
    weights = np.asarray([.5 + (i % 7) * .25 for i in range(len(x))], dtype=np.float32)
    cases = []
    for name, extra in specifications():
        started = time.monotonic()
        params = {**BASE, **extra}
        data = lgb.Dataset(x, label=y, weight=weights, params=params)
        model = lgb.train(params, data, num_boost_round=4)
        filename = f'models/constraints_extra/{name}.txt'
        model.save_model(str(fixtures / filename))
        cases.append(dict(name=name, params=params, model_file=filename,
                          training=dict(rows=x.tolist(), labels=y.tolist(), weights=weights.tolist(), num_boost_round=4),
                          rows=x.tolist(), raw=np.asarray(model.predict(x, raw_score=True)).reshape(-1, 1).tolist(),
                          leaves=np.asarray(model.predict(x, pred_leaf=True)).tolist()))
        check_resources(started)
    assert {c['name'] for c in cases} == NAMES
    save_json(fixtures / 'prediction_constraints_extra.json', dict(schema_version=1, upstream_commit=COMMIT,
              library=record['library'], compiler=record['compiler'], compile_commands_sha256=record['compile_commands_sha256'],
              producer_sha256=sha(ROOT / 'tools/oracle/generate_constraints_extra.py'), cases=cases))
    save_json(fixtures / 'prediction_constraints_extra.schema.json', schema())
    print(f'Generated {len(cases)} additional actual native learner-control cases')


if __name__ == '__main__':
    generate()
