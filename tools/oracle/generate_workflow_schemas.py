"""Write strict saved-evidence schemas for workflow and model-operation cases."""
from common import COMMIT, ROOT, save_json


def array(items, minimum=0):
    return dict(type='array', items=items, minItems=minimum)


def obj(properties, required=None, extra=False):
    return dict(type='object', properties=properties, required=list(properties) if required is None else required, additionalProperties=extra)


def ref(name):
    return {'$ref': '#/$defs/' + name}


NUMBER = dict(type='number')
INTEGER = dict(type='integer', minimum=0)
BOOLEAN = dict(type='boolean')
STRING = dict(type='string', minLength=1)
NULL = dict(type='null')
MATRIX = array(array(NUMBER), 1)
NULLABLE = lambda value: dict(anyOf=[value, NULL])
PARAMS = dict(type='object', minProperties=1)
ERROR = dict(error_type=STRING, error=STRING)
DEFS = dict(
    metric=obj(dict(dataset_name=STRING, metric_name=STRING, metric_value=NUMBER, maximize=BOOLEAN,
                    metric_std_dev=NULLABLE(NUMBER))),
    model=obj(dict(model_file=dict(type='string', pattern=r'^models/[a-z0-9_]+\.txt$'), current_iteration=INTEGER,
                   num_trees=INTEGER, raw=MATRIX, transformed=MATRIX, leaves=array(array(INTEGER), 1))),
    snapshot=obj(dict(current_iteration=INTEGER, num_trees=INTEGER, raw=MATRIX, transformed=MATRIX, leaves=array(array(INTEGER), 1))),
    provenance=obj(dict(upstream_commit=dict(const=COMMIT), loaded_library_sha256=dict(type='string', pattern='^[a-f0-9]{64}$'),
                        python_sources=obj({key: dict(type='string', pattern='^[a-f0-9]{64}$') for key in ('engine.py', 'callback.py', 'basic.py', 'compat.py')}),
                        dependencies=obj({key: STRING for key in ('numpy', 'scipy', 'scikit-learn')}), execution=STRING, iteration_index=STRING)),
    history=dict(type='object', additionalProperties=dict(type='object', additionalProperties=array(NUMBER))),
    folds=array(obj(dict(train_indices=array(INTEGER, 1), valid_indices=array(INTEGER, 1))), 2),
    params=PARAMS,
    dump_tree=dict(type='object', properties=dict(split_gain=dict(anyOf=[NUMBER, dict(enum=['Infinity', '-Infinity', 'NaN'])]),
                                                 left_child=ref('dump_tree'), right_child=ref('dump_tree'))),
    original_training=obj(dict(rows=MATRIX, labels=array(NUMBER, 1), weights=array(NUMBER, 1), groups=NULLABLE(array(INTEGER, 1)),
                               positions=NULLABLE(array(dict(type='integer'), 1)), num_boost_round=dict(type='integer', minimum=1)),
                          required=['rows', 'labels', 'num_boost_round']),
)


def document(case):
    return dict({'$schema': 'https://json-schema.org/draft/2020-12/schema', '$defs': DEFS},
                **obj(dict(schema_version=dict(const=1), upstream_commit=dict(const=COMMIT), provenance=ref('provenance'), cases=array(case, 1))))


def schemas():
    evaluations = array(ref('metric'))
    event = obj(dict(callback=STRING, phase=dict(enum=['enter', 'exit', 'raise']), iteration=INTEGER, begin_iteration=INTEGER,
                     end_iteration=dict(type='integer', minimum=1), model_current_iteration=INTEGER, model_best_iteration=INTEGER,
                     learning_rate=NUMBER, evaluation=NULLABLE(evaluations),
                     history_lengths=dict(type='object', additionalProperties=dict(type='object', additionalProperties=INTEGER)),
                     enabled=BOOLEAN, best_iterations_zero=array(INTEGER), exception_best_iteration_zero=INTEGER,
                     exception_best_score=evaluations, **ERROR),
                required=['callback', 'phase', 'iteration', 'begin_iteration', 'end_iteration', 'model_current_iteration',
                          'model_best_iteration', 'learning_rate', 'evaluation', 'history_lengths'])
    workflow = obj(dict(name=STRING, params=ref('params'), rows=MATRIX, labels=array(NUMBER, 1),
                        valid_sets=array(obj(dict(name=STRING, is_training=BOOLEAN, labels=array(NUMBER, 1)))),
                        num_boost_round=dict(type='integer', minimum=1),
                        stopping=NULLABLE(obj(dict(stopping_rounds=dict(type='integer', minimum=1), first_metric_only=BOOLEAN,
                                                   verbose=BOOLEAN, min_delta=dict(anyOf=[NUMBER, array(NUMBER)])))),
                        learning_rate_schedule=NULLABLE(array(NUMBER, 1)),
                        callback_order=array(obj(dict(name=STRING, order=dict(type='integer'), before_iteration=BOOLEAN)), 1),
                        events=array(event, 1), history=ref('history'), warnings=array(STRING),
                        derivatives=array(obj(dict(raw_input=array(NUMBER, 1), gradients=array(NUMBER, 1), hessians=array(NUMBER, 1)))),
                        best_iteration=INTEGER, best_score=dict(type='object', additionalProperties=dict(type='object', additionalProperties=NUMBER)),
                        live_model=ref('snapshot'), final_model=ref('model'), full_model=ref('model'), builtin_reference=ref('model'),
                        builtin_evaluation=evaluations,
                        error_layer=STRING, **ERROR),
                   required=['name', 'params', 'rows', 'labels', 'valid_sets', 'num_boost_round', 'stopping', 'learning_rate_schedule',
                             'callback_order', 'events', 'history', 'warnings', 'derivatives'])
    workflow['oneOf'] = [dict(required=['error_type', 'error', 'error_layer']), dict(required=['best_iteration', 'best_score', 'live_model', 'final_model', 'full_model'])]
    cv = obj(dict(name=STRING, params=ref('params'), rows=MATRIX, labels=array(NUMBER, 1), groups=NULLABLE(array(INTEGER, 1)),
                  num_boost_round=dict(type='integer', minimum=1), seed=INTEGER, nfold=dict(type='integer', minimum=2), stratified=BOOLEAN,
                  shuffle=BOOLEAN, fold_generator=STRING, permutation=NULLABLE(array(INTEGER, 1)), source_folds=ref('folds'), folds=ref('folds'),
                  aggregate_std=dict(const='population; original engine numpy.std ddof=0'),
                  steps=array(obj(dict(iteration=INTEGER, aggregate=evaluations,
                                       folds=array(obj(dict(evaluation=evaluations, current_iteration=INTEGER)), 2))), 1),
                  history=dict(type='object', minProperties=1, additionalProperties=array(NUMBER, 1)),
                  stopping=NULLABLE(obj(dict(stopping_rounds=dict(type='integer', minimum=1), min_delta=NUMBER))),
                  best_iteration=dict(type='integer', minimum=-1), cvbooster_file=dict(type='string', pattern=r'^models/cv_[a-z_]+_booster\.json$'),
                  models=array(obj(dict(best_iteration=dict(type='integer', minimum=-1), final_model=ref('model'), full_model=ref('model'))), 2)))
    op_properties = dict(name=STRING, operation=dict(enum=['continuation', 'initial_scores', 'refit', 'rollback', 'set_leaf_output',
                                                          'prediction_early_stop', 'extreme_gain_diagnostic']),
                         params=ref('params'), effective_params=ref('params'),
                         backend_params=obj(dict(boosting=dict(enum=['gbdt', 'dart', 'rf']), objective=STRING,
                                                 data_sample_strategy=dict(enum=['bagging', 'goss']), bagging_by_query=BOOLEAN,
                                                 refit_decay_rate=dict(type='number', minimum=0, maximum=1))),
                         rows=MATRIX, labels=array(NUMBER, 1), original_training=ref('original_training'),
                         initial_model=ref('model'), final_model=ref('model'), supplied_initial_scores=NULLABLE(array(NUMBER, 1)),
                         initial_scores=NULLABLE(array(NUMBER, 1)), num_boost_round=dict(type='integer', minimum=1),
                         warnings=array(STRING), training_score_buffer=array(NUMBER, 1),
                         weights=array(NUMBER, 1), groups=NULLABLE(array(INTEGER, 1)), positions=NULLABLE(array(dict(type='integer'), 1)),
                         decay_rate=dict(type='number', enum=[-.1, 0., .5, .9, 1., 1.1]), leaf_pred=array(array(INTEGER, 1), 1),
                         refit_steps=array(obj(dict(iteration=INTEGER, training_scores_before=array(NUMBER, 1),
                                                   gradients=array(NUMBER, 1), hessians=array(NUMBER, 1))), 1),
                         training_scores_after=array(NUMBER, 1), tree_id=INTEGER, leaf_id=INTEGER, old_output=NUMBER,
                         set_output=NUMBER, get_output=NUMBER, leaves_before=array(array(INTEGER, 1), 1),
                         probes=array(obj(dict(params=ref('params'), raw=MATRIX, transformed=MATRIX, leaves=array(array(INTEGER), 1),
                                               contributions=MATRIX, **ERROR), required=['params']), 1),
                         model_dump=dict(type='object', required=['tree_info'], properties=dict(
                             tree_info=array(dict(type='object', required=['tree_structure'], properties=dict(tree_structure=ref('dump_tree'))), 1))), contributions=MATRIX,
                         importance_split=array(INTEGER, 1),
                         importance_gain=array(dict(anyOf=[NUMBER, dict(enum=['Infinity', '-Infinity', 'NaN'])]), 1),
                         prefix_raw=array(MATRIX, 1),
                         nonfinite_encoding=dict(const='Infinity, -Infinity, NaN strings represent actual nonfinite native outputs'), **ERROR)
    continuation_step = obj(dict(iteration=INTEGER, begin_iteration=INTEGER, end_iteration=INTEGER,
                                 training_score_buffer=array(NUMBER, 1), model=ref('snapshot')))
    rollback_step = obj(dict(call=INTEGER, before_iteration=INTEGER, model=ref('model')))
    op_properties['steps'] = array(dict(oneOf=[continuation_step, rollback_step]))
    operations = obj(op_properties, required=['name', 'operation', 'params', 'rows', 'labels'])
    requirements = dict(continuation=['original_training', 'initial_model', 'supplied_initial_scores', 'num_boost_round', 'steps', 'warnings'],
                        initial_scores=['initial_scores', 'num_boost_round', 'training_score_buffer', 'final_model'],
                        refit=['initial_model', 'original_training', 'weights', 'groups', 'positions', 'initial_scores', 'decay_rate', 'leaf_pred'],
                        rollback=['initial_model', 'steps', 'num_boost_round'],
                        set_leaf_output=['initial_model', 'tree_id', 'leaf_id', 'old_output', 'set_output', 'get_output', 'leaves_before', 'final_model'],
                        prediction_early_stop=['initial_model', 'num_boost_round', 'probes', 'prefix_raw'],
                        extreme_gain_diagnostic=['final_model', 'num_boost_round', 'model_dump', 'contributions', 'importance_split', 'importance_gain', 'nonfinite_encoding'])
    operations['allOf'] = [dict(**{'if': dict(properties=dict(operation=dict(const=kind)))}, then=dict(required=fields)) for kind, fields in requirements.items()]
    operations['allOf'].append(dict(**{'if': dict(properties=dict(operation=dict(const='refit')))}, then=dict(oneOf=[
        dict(required=['error_type', 'error'], **{'not': dict(required=['final_model'])}),
        dict(required=['final_model', 'effective_params', 'refit_steps', 'training_scores_after'],
             properties=dict(decay_rate=dict(type='number', minimum=0, maximum=1)), **{'not': dict(required=['error'])})])))
    return dict(workflow=document(workflow), cv=document(cv), prediction_operations=document(operations))


def generate(fixtures=ROOT / 'tests/fixtures'):
    for name, schema in schemas().items():
        save_json(fixtures / f'{name}.schema.json', schema)


if __name__ == '__main__':
    generate()
