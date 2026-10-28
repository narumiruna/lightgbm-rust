"""Strict schemas for final native workflow, AUC-mu, and empty-query evidence."""
import copy

from common import COMMIT, ROOT, save_json
from check_final import NAMES, SORT_HEADERS
from generate_workflow_schemas import DEFS, NUMBER, INTEGER, BOOLEAN, STRING, MATRIX, NULLABLE, array, obj, ref

DIGEST = dict(type='string', pattern='^[a-f0-9]{64}$')
VECTOR = array(NUMBER, 1)


def document(family, properties):
    properties = dict(schema_version=dict(const=1), upstream_commit=dict(const=COMMIT), **properties)
    properties['cases']['minItems'] = properties['cases']['maxItems'] = len(NAMES[family])
    return dict({'$schema': 'https://json-schema.org/draft/2020-12/schema', '$defs': copy.deepcopy(DEFS)}, **obj(properties))


def schemas():
    evaluation = array(ref('metric'), 1)
    shared = dict(name=dict(enum=sorted(NAMES['workflow_extended'])), kind=dict(enum=['train', 'cv']),
                  params=ref('params'), rows=MATRIX, labels=VECTOR, learning_rates=VECTOR, schedule_after=BOOLEAN)
    train = obj(dict(shared, kind=dict(const='train'), initial_model=NULLABLE(ref('model')),
                     steps=array(obj(dict(iteration=INTEGER, begin_iteration=INTEGER, end_iteration=INTEGER,
                                          learning_rate=NUMBER, evaluation=evaluation, training_scores=VECTOR, raw=VECTOR)), 1),
                     final_model=ref('model')))
    cv = obj(dict(shared, kind=dict(const='cv'), custom=BOOLEAN, stopping=BOOLEAN, folds=ref('folds'),
                  objective_calls=array(obj(dict(rows=dict(type='integer', minimum=1), predictions=VECTOR))),
                  trace=array(obj(dict(phase=dict(enum=['before', 'after_early', 'after_record']), iteration=INTEGER,
                                       history_length=INTEGER)), 1),
                  steps=array(obj(dict(iteration=INTEGER, learning_rate=NUMBER, evaluation=evaluation,
                                       folds=array(obj(dict(evaluation=evaluation, raw=VECTOR)), 3))), 1),
                  history=dict(type='object', minProperties=4, maxProperties=4, additionalProperties=VECTOR),
                  best_iteration=dict(type='integer', minimum=-1), models=array(ref('model'), 3)))
    workflow = document('workflow_extended', dict(provenance=ref('provenance'), cases=array(dict(oneOf=[train, cv]), 1)))
    metric_case = obj(dict(name=dict(enum=sorted(NAMES['metric_edges'])), objective=dict(const='multiclass'),
                           metric=dict(const='auc_mu'), params=ref('params'), labels=VECTOR, scores=VECTOR,
                           weights=VECTOR, score_bits=array(dict(type='string', pattern='^[a-f0-9]{16}$'), 1),
                           results=dict(type='array', minItems=1, maxItems=1, items=obj(dict(
                               name=dict(const='auc_mu'), higher_is_better=dict(const=True),
                               value=dict(type='number', minimum=0, maximum=1))))),
                      required=['name', 'objective', 'metric', 'params', 'labels', 'scores', 'score_bits', 'results'])
    metric = document('metric_edges', dict(
        compiler_version=STRING, selected_toolchain_sha256=DIGEST, sort_header_sha256=obj({name: DIGEST for name in SORT_HEADERS}),
        loaded_library_sha256=DIGEST, loaded_library_path=STRING, native_driver_source_sha256=DIGEST,
        native_driver_sha256=DIGEST, generator_sha256=DIGEST,
        sort_contract=dict(const='single-thread Apple clang 21.0.0 / libc++ 2200.27.0 std::sort; EPS comparison is not strict weak ordering'),
        epsilon_bits=dict(const='3cd203afa0000000'), absolute_bound=dict(const=1e-12), cases=array(metric_case, 1)))
    metric_result = obj(dict(name=STRING, value=dict(type='number', minimum=0, maximum=1)))
    step = obj(dict(scores=VECTOR, queries=array(INTEGER), gradients=VECTOR, hessians=VECTOR,
                    metrics=dict(type='array', minItems=10, maxItems=10, items=metric_result)),
               required=['scores', 'gradients', 'hessians', 'metrics'])
    training = obj(dict(params=ref('params'), rounds=dict(const=3), predictions=dict(type='array', minItems=3, maxItems=3, items=VECTOR),
                        metrics=dict(type='object', minProperties=10, maxProperties=10,
                                     additionalProperties=dict(type='array', minItems=3, maxItems=3, items=NUMBER)),
                        trees=dict(type='integer', minimum=1, maximum=3)))
    empty_case = obj(dict(name=dict(enum=sorted(NAMES['empty_queries'])), groups=array(INTEGER, 1), boundaries=array(INTEGER, 2),
                          labels=VECTOR, positions=array(dict(type='integer'), 1),
                          steps=dict(type='array', minItems=4, maxItems=4, items=step), training=training, query_bagging=training),
                     required=['name', 'groups', 'boundaries', 'labels', 'steps', 'training'])
    empty = document('empty_queries', dict(loaded_library_sha256=DIGEST, native_driver_sha256=DIGEST,
        native_driver_source_sha256=DIGEST, generator_sha256=DIGEST,
        source_sha256=obj({name: DIGEST for name in ('src/io/metadata.cpp', 'src/objective/rank_objective.hpp',
                                                    'src/metric/rank_metric.hpp', 'src/metric/map_metric.hpp', 'src/boosting/bagging.hpp')}),
        restrictions=obj({name: STRING for name in ('lambdarank_empty', 'weighted_empty', 'query_bagging_queries_exceed_rows', 'row_subset_empty')}),
        cases=array(empty_case, 1)))
    return dict(workflow_extended=workflow, metric_edges=metric, empty_queries=empty)


def generate(fixtures=ROOT / 'tests/fixtures'):
    for name, schema in schemas().items():
        save_json(fixtures / f'{name}.schema.json', schema)


if __name__ == '__main__':
    generate()
