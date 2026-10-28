"""Check final native families, including exact inputs and stateful call order.

These checks do not compute objective or metric substitutes. Reference mode
runs the actual producers; only explicitly recorded relink hashes are removed.
"""
import copy
import hashlib
import json
import math
from pathlib import Path
import shlex
import struct

from common import COMMIT, OUT, ROOT, SOURCE, sha
from check_workflows import close, model_shape
from generate_predictions import BASE
from generate_workflow_extended import NAMES as WORKFLOW_NAMES

EMPTY_GROUPS = dict(nonempty=[2, 2], leading=[0, 2, 2], middle=[2, 0, 2],
                    trailing=[2, 2, 0], repeated=[0, 0, 2, 0, 0, 2, 0, 0],
                    short=[0, 1, 0, 2, 1, 0], one_row=[0, 1, 0, 0])
EMPTY_NAMES = {name + suffix for name in EMPTY_GROUPS for suffix in ('', '_positions')}
SORT_HEADERS = ('sort.h', 'make_heap.h', 'sift_down.h', 'pop_heap.h', 'push_heap.h', 'sort_heap.h')


def metric_specs():
    # Inputs only. All expected metric values come from the native factory.
    from generate_metric_edges import case
    cases = [case(1, 128)]
    for seed in (1, 7, 42, 2026):
        for rows in (5, 23, 24, 127, 129, 257, 1025):
            for weighted in (False, True):
                cases.append(case(seed, rows, 3 if seed == 42 else 2, weighted, rows % 2 == 1))
    for seed in (1, 7, 42, 2026):
        for weighted in (False, True):
            if seed != 1 or weighted:
                cases.append(case(seed, 128, 2, weighted))
    epsilon = struct.unpack('>f', struct.pack('>f', 1e-15))[0]
    boundaries = [0., epsilon / 2, math.nextafter(epsilon / 2, 0.),
                  math.nextafter(epsilon / 2, math.inf), epsilon, 1.5 * epsilon, -epsilon / 2, -epsilon]
    for weighted in (False, True):
        c = case(42, 64, 2, weighted)
        c['name'] += '_exact_epsilon_boundaries'
        c['scores'][:64] = [boundaries[i % len(boundaries)] for i in range(64)]
        cases.append(c)
        c = case(42, 129, 3, weighted)
        c['name'] += '_asymmetric_matrix'
        c['params']['auc_mu_weights'] = [0., 2., .5, 1., 0., 3., 4., .25, 0.]
        cases.append(c)
    return {c['name']: c for c in cases}


NAMES = dict(workflow_extended=WORKFLOW_NAMES, metric_edges=set(metric_specs()), empty_queries=EMPTY_NAMES)


def stable_payload(family, data):
    result = copy.deepcopy(data)
    if family != 'workflow_extended':
        del result['native_driver_sha256']
    return result


def exact_payload(family, actual, expected):
    assert stable_payload(family, actual) == stable_payload(family, expected), f'{family} deterministic payload changed'


def safe_relative(path):
    value = Path(path)
    assert not value.is_absolute() and '..' not in value.parts and str(value) == path, 'unsafe relative path'


def coverage(family, data):
    names = [c['name'] for c in data['cases']]
    assert len(names) == len(set(names)) and set(names) == NAMES[family], f'{family} missing or unexpected case'


def source_identity(family, data, manifest):
    if family == 'workflow_extended':
        p = data['provenance']
        assert p['upstream_commit'] == COMMIT
        assert p['loaded_library_sha256'] == manifest['loaded_library']['loaded_sha256']
        hashes = {r['path']: r['sha256'] for r in manifest['reference_sources']}
        assert p['python_sources'] == {name: hashes['python-package/lightgbm/' + name]
                                      for name in ('engine.py', 'callback.py', 'basic.py', 'compat.py')}
        assert p['dependencies'] == dict(numpy='2.2.6', scipy='1.15.3', **{'scikit-learn': '1.6.1'})
        return
    assert data['generator_sha256'] == sha(ROOT / f'tools/oracle/generate_{family}.py')
    assert data['loaded_library_sha256'] == manifest['loaded_library']['loaded_sha256']
    helper = manifest['family_builds'][family]
    assert helper['loaded_library_sha256'] == data['loaded_library_sha256']
    assert helper['source_sha256'] == data['native_driver_source_sha256']
    assert helper['shared_header_sha256'] == sha(ROOT / 'tools/oracle/driver_common.hpp')
    assert helper['compile_commands_sha256'] == manifest['build']['compile_commands_sha256']
    paths = [entry['path'] for entry in helper['object_sha256']]
    assert len(paths) == len(set(paths)) and paths
    for path in paths:
        safe_relative(path)
    if family == 'metric_edges':
        assert data['loaded_library_path'] == manifest['loaded_library']['loaded_path']
        assert data['native_driver_source_sha256'] == sha(ROOT / 'tools/oracle/extended_driver.cpp')
        assert data['compiler_version'] == manifest['build']['compiler']['version']
        assert data['selected_toolchain_sha256'] == manifest['build']['compiler']['selected_toolchain_sha256']
        assert data['sort_header_sha256'] == manifest['sort_headers']['sha256']
        assert data['epsilon_bits'] == '3cd203afa0000000' and data['absolute_bound'] == 1e-12
    else:
        from generate_empty_queries import DRIVER
        assert data['native_driver_source_sha256'] == hashlib.sha256(DRIVER.encode()).hexdigest()
        hashes = {r['path']: r['sha256'] for r in manifest['reference_sources']}
        assert data['source_sha256'] == {p: hashes[p] for p in (
            'src/io/metadata.cpp', 'src/objective/rank_objective.hpp',
            'src/metric/rank_metric.hpp', 'src/metric/map_metric.hpp', 'src/boosting/bagging.hpp')}


def workflow(data):
    for c in data['cases']:
        name, params = c['name'], c['params']
        rows = [[i - 12, (i * 5) % 11 - 5] for i in range(24)]
        labels = [2 * a - b + (3 if a > 0 else -2) for a, b in rows]
        if name == 'dart_continue_original_three':
            labels = [v + (2 if i % 2 else -1) for i, v in enumerate(labels)]
        assert c['rows'] == rows and c['labels'] == labels
        n = len(rows)
        assert c['schedule_after'] == name.endswith('_after')
        rates = [BASE['learning_rate']] * 3 if name == 'dart_continue_original_three' else [.25, .25, .4, .4, .15, .15, .3, .3]
        assert c['learning_rates'] == rates
        expected = dict(BASE, objective='regression')
        if c['kind'] == 'train':
            expected.update(boosting='dart', metric=['l2', 'l1'], drop_rate=.7, skip_drop=.2, max_drop=3,
                            drop_seed=7, uniform_drop=name == 'dart_schedule_uniform',
                            xgboost_dart_mode=name == 'dart_continue_xgboost')
            if name.startswith('goss_schedule'):
                expected.update(boosting='gbdt', data_sample_strategy='goss', top_rate=.6, other_rate=.2)
            if name.startswith('gbdt_schedule'):
                expected['boosting'] = 'gbdt'
            if name.endswith('_features'):
                expected['feature_fraction'] = .5
            if name == 'dart_continue_original_three':
                expected.update(skip_drop=0., drop_rate=.5, drop_seed=0, max_drop=50)
            assert params == expected
            initial = c['initial_model']
            begin = 4 if name == 'dart_continue_original_three' else 5 if name.startswith('dart_continue') else 0
            assert (initial is None) == (begin == 0)
            if initial:
                model_shape(initial, rows, 1)
                assert initial['current_iteration'] == begin
            assert len(c['steps']) == len(rates)
            for i, step in enumerate(c['steps']):
                assert step['iteration'] == begin + i and step['begin_iteration'] == begin
                assert step['end_iteration'] == begin + len(rates)
                assert len(step['raw']) == len(step['training_scores']) == n
                effective_rate = (BASE['learning_rate'] if i == 0 else rates[i - 1]) if c['schedule_after'] else rates[i]
                assert step['learning_rate'] == effective_rate
                assert [(v['dataset_name'], v['metric_name']) for v in step['evaluation']] == [
                    (ds, metric) for ds in ('train', 'valid') for metric in ('l2', 'l1')]
                assert all(not v['maximize'] and v['metric_std_dev'] is None for v in step['evaluation'])
                # GOSS's score buffer includes its native sampled update state;
                # it is not an alias for model.predict on all training rows.
                if not name.startswith('goss'):
                    for a, b in zip(step['raw'], step['training_scores']):
                        close(a, b)
            model_shape(c['final_model'], rows, 1)
            assert c['final_model']['current_iteration'] == begin + len(rates)
            for actual, expected_raw in zip(c['final_model']['raw'], c['steps'][-1]['raw']):
                close(actual[0], expected_raw)
        else:
            expected.update(metric='None' if c['custom'] else 'l2', boost_from_average=False,
                            objective='custom' if c['custom'] else 'regression')
            assert params == expected
            assert c['custom'] == name.startswith('cv_custom') and c['stopping'] == name.endswith('_stop')
            count = 3 if c['stopping'] else 8
            assert len(c['steps']) == count and c['best_iteration'] == (1 if c['stopping'] else -1)
            assert c['trace'] == [dict(phase=phase, iteration=i, history_length=i + (phase == 'after_record'))
                                  for i in range(count) for phase in ('before', 'after_early', 'after_record')]
            assert c['folds'] == [dict(train_indices=[i for i in range(n) if i % 3 != k],
                                       valid_indices=[i for i in range(n) if i % 3 == k]) for k in range(3)]
            assert len(c['models']) == 3
            for model in c['models']:
                model_shape(model, rows, 1)
                assert model['current_iteration'] == count
            metric = 'custom_l2' if c['custom'] else 'l2'
            history_count = 1 if c['stopping'] else count
            assert set(c['history']) == {ds + ' ' + metric + suffix for ds in ('train', 'valid') for suffix in ('-mean', '-stdv')}
            assert all(len(values) == history_count for values in c['history'].values())
            assert len(c['objective_calls']) == (count * 3 if c['custom'] else 0)
            for i, step in enumerate(c['steps']):
                assert step['iteration'] == i and len(step['folds']) == 3
                effective_rate = (BASE['learning_rate'] if i == 0 else rates[i - 1]) if c['schedule_after'] else rates[i]
                assert step['learning_rate'] == effective_rate
                assert len(step['evaluation']) == 2
                for k, fold in enumerate(step['folds']):
                    assert len(fold['raw']) == n and len(fold['evaluation']) == 2
                    if c['custom']:
                        call = c['objective_calls'][i * 3 + k]
                        assert call['rows'] == 16 and len(call['predictions']) == 16
                        previous = c['steps'][i - 1]['folds'][k]['raw'] if i else [0.] * n
                        for a, b in zip(call['predictions'], [previous[r] for r in c['folds'][k]['train_indices']]):
                            close(a, b)
                for j, aggregated in enumerate(step['evaluation']):
                    assert (aggregated['dataset_name'], aggregated['metric_name'], aggregated['maximize']) == (('train', 'valid')[j], metric, False)
                    values = []
                    for fold in step['folds']:
                        result = fold['evaluation'][j]
                        assert (result['dataset_name'], result['metric_name'], result['maximize'], result['metric_std_dev']) == (('train', 'valid')[j], metric, False, None)
                        values.append(result['metric_value'])
                    mean = sum(values) / 3
                    close(aggregated['metric_value'], mean)
                    close(aggregated['metric_std_dev'], math.sqrt(sum((v - mean) ** 2 for v in values) / 3))
                    if i < history_count:
                        prefix = aggregated['dataset_name'] + ' ' + metric
                        assert c['history'][prefix + '-mean'][i] == aggregated['metric_value']
                        assert c['history'][prefix + '-stdv'][i] == aggregated['metric_std_dev']
            for k, model in enumerate(c['models']):
                for a, b in zip(model['raw'], c['steps'][-1]['folds'][k]['raw']):
                    close(a[0], b)


def metric_edges(data):
    specifications = metric_specs()
    for c in data['cases']:
        expected = specifications[c['name']]
        for key, value in expected.items():
            assert c[key] == value, (c['name'], key)
        n, classes = len(c['labels']), c['params']['num_class']
        assert len(c['scores']) == n * classes
        assert c['score_bits'] == [struct.pack('>d', v).hex() for v in c['scores']]
        assert c['results'][0]['name'] == 'auc_mu' and c['results'][0]['higher_is_better'] is True
        assert len(c['results']) == 1 and 0 <= c['results'][0]['value'] <= 1
    assert data['cases'][0]['results'][0]['value'] == .5346219721066797


def empty_queries(data):
    metric_names = [f'{metric}@{i}' for metric in ('ndcg', 'map') for i in range(1, 6)]
    params = dict(objective='rank_xendcg', objective_seed=7, metric=['ndcg', 'map'],
                  num_threads=1, verbosity=-1, min_data_in_leaf=1, min_data_in_bin=1,
                  feature_pre_filter=False, enable_bundle=False, num_leaves=3,
                  learning_rate=.1, deterministic=True, force_col_wise=True)
    for c in data['cases']:
        with_positions = c['name'].endswith('_positions')
        name = c['name'].removesuffix('_positions')
        groups = EMPTY_GROUPS[name]
        assert c['groups'] == groups
        n = sum(groups)
        assert c['labels'] == [0., 1., 1., 0.][:n]
        boundaries = [0]
        for size in groups:
            boundaries.append(boundaries[-1] + size)
        assert c['boundaries'] == boundaries
        assert ('positions' in c) == with_positions
        if with_positions:
            assert c['positions'] == [9, -7, 9, -7][:n]
        assert len(c['steps']) == 4
        scores = [.4, -.2, .1, .8][:n]
        for i, step in enumerate(c['steps']):
            assert step['scores'] == (list(reversed(scores)) if i == 1 else scores)
            assert len(step['gradients']) == len(step['hessians']) == n
            assert all(v >= 0 for v in step['hessians'])
            assert ('queries' in step) == (i == 2)
            if i == 2:
                assert step['queries'] == [q for q in range(len(groups)) if q % 2 == 0]
                selected = {r for q in step['queries'] for r in range(boundaries[q], boundaries[q + 1])}
                for r in set(range(n)) - selected:
                    assert step['gradients'][r] == c['steps'][1]['gradients'][r]
                    assert step['hessians'][r] == c['steps'][1]['hessians'][r]
            assert [v['name'] for v in step['metrics']] == metric_names
            assert all(0 <= v['value'] <= 1 for v in step['metrics'])
            for start, end in zip(boundaries, boundaries[1:]):
                if end - start == 1:
                    assert step['gradients'][start] == step['hessians'][start] == 0
        assert ('query_bagging' in c) == (len(groups) <= n)
        for field in ('training', 'query_bagging'):
            if field not in c:
                continue
            training = c[field]
            assert training['rounds'] == 3 and len(training['predictions']) == 3
            assert 1 <= training['trees'] <= 3
            assert all(len(row) == n and all(math.isfinite(v) for v in row) for row in training['predictions'])
            assert set(training['metrics']) == set(metric_names)
            assert all(len(v) == 3 and all(0 <= score <= 1 for score in v) for v in training['metrics'].values())
            expected_params = params if field == 'training' else dict(params, bagging_by_query=True,
                bagging_freq=1, bagging_fraction=.75, bagging_seed=3)
            assert training['params'] == expected_params
    # Position metadata is ignored by native XE; it must not change its RNG.
    cases = {c['name']: c for c in data['cases']}
    for name in EMPTY_GROUPS:
        a, b = copy.deepcopy(cases[name]), copy.deepcopy(cases[name + '_positions'])
        del a['name'], b['name'], b['positions']
        assert a == b, f'XE position/RNG grouping changed: {name}'


def validate(family, data, manifest=None):
    assert data['schema_version'] == 1 and data['upstream_commit'] == COMMIT
    coverage(family, data)
    if manifest:
        source_identity(family, data, manifest)
    {'workflow_extended': workflow, 'metric_edges': metric_edges, 'empty_queries': empty_queries}[family](data)


def self_test(must_fail):
    import jsonschema
    from generate_final_schemas import schemas
    manifest = json.loads((ROOT / 'docs/reference-manifest.json').read_text())
    for family, schema in schemas().items():
        original = json.loads((ROOT / f'tests/fixtures/{family}.json').read_text())
        changed = copy.deepcopy(original)
        changed['cases'].pop()
        must_fail(f'{family} missing case', lambda: coverage(family, changed))
        changed = copy.deepcopy(original)
        if family == 'workflow_extended':
            changed['cases'][0]['steps'][0]['raw'][0] += .01
        elif family == 'metric_edges':
            changed['cases'][0]['results'][0]['value'] += .01
        else:
            changed['cases'][0]['steps'][0]['gradients'][0] += .01
        must_fail(f'{family} changed native value', lambda: exact_payload(family, changed, original))
        changed = copy.deepcopy(original)
        if family == 'metric_edges':
            changed['cases'][0]['scores'].pop()
        else:
            changed['cases'][0]['steps'][0]['raw' if family == 'workflow_extended' else 'gradients'].pop()
        must_fail(f'{family} malformed shape', lambda: validate(family, changed, manifest))
        changed = copy.deepcopy(original)
        changed['cases'][0]['unexpected'] = 1
        must_fail(f'{family} unexpected field', lambda: jsonschema.Draft202012Validator(schema).validate(changed))
        changed = copy.deepcopy(original)
        if family == 'workflow_extended':
            changed['provenance']['python_sources']['engine.py'] = '0' * 64
        else:
            changed['native_driver_source_sha256'] = '0' * 64
        must_fail(f'{family} source identity', lambda: source_identity(family, changed, manifest))
    original = json.loads((ROOT / 'tests/fixtures/metric_edges.json').read_text())
    changed = copy.deepcopy(original)
    changed['cases'][0]['score_bits'][0] = '0' * 16
    must_fail('AUC-mu wrong IEEE score bits', lambda: metric_edges(changed))
    changed = copy.deepcopy(original)
    changed['sort_header_sha256']['sort.h'] = '0' * 64
    must_fail('AUC-mu sort header hash', lambda: source_identity('metric_edges', changed, manifest))
    changed = copy.deepcopy(original)
    changed['selected_toolchain_sha256'] = '0' * 64
    must_fail('AUC-mu compiler identity', lambda: source_identity('metric_edges', changed, manifest))
    original = json.loads((ROOT / 'tests/fixtures/empty_queries.json').read_text())
    changed = copy.deepcopy(original)
    case = next(c for c in changed['cases'] if c['name'] == 'leading')
    case['groups'] = [2, 0, 2]
    case['boundaries'] = [0, 2, 2, 4]
    must_fail('empty query zero-group location', lambda: empty_queries(changed))
    changed = copy.deepcopy(original)
    case = next(c for c in changed['cases'] if c['name'] == 'repeated_positions')
    case['positions'][0] = -7
    must_fail('empty query position metadata', lambda: empty_queries(changed))
    changed = copy.deepcopy(original)
    case = next(c for c in changed['cases'] if c['name'] == 'leading_positions')
    case['steps'][3]['gradients'] = case['steps'][0]['gradients'][:]
    must_fail('empty query stateful RNG grouping', lambda: empty_queries(changed))
    changed = copy.deepcopy(original)
    case = next(c for c in changed['cases'] if c['name'] == 'leading')
    case['steps'][2]['queries'] = [1]
    must_fail('empty query sampled call order', lambda: empty_queries(changed))
    original = json.loads((ROOT / 'tests/fixtures/workflow_extended.json').read_text())
    changed = copy.deepcopy(original)
    changed['cases'][0]['steps'][2]['learning_rate'] = .25
    must_fail('extended workflow schedule order', lambda: workflow(changed))
    changed = copy.deepcopy(original)
    case = next(c for c in changed['cases'] if c['name'] == 'cv_custom_schedule')
    case['objective_calls'][3]['predictions'][0] += .01
    must_fail('extended CV derivative input state', lambda: workflow(changed))
    changed = copy.deepcopy(original)
    case = next(c for c in changed['cases'] if c['name'] == 'cv_custom_schedule')
    case['steps'][0]['evaluation'][0]['metric_std_dev'] += .01
    must_fail('extended CV population deviation', lambda: workflow(changed))
    for family in ('metric_edges', 'empty_queries'):
        data = json.loads((ROOT / f'tests/fixtures/{family}.json').read_text())
        for key in ('source_sha256', 'shared_header_sha256', 'compile_commands_sha256', 'loaded_library_sha256'):
            changed_manifest = copy.deepcopy(manifest)
            changed_manifest['family_builds'][family][key] = '0' * 64
            must_fail(f'{family} helper {key}', lambda: source_identity(family, data, changed_manifest))
        changed_manifest = copy.deepcopy(manifest)
        del changed_manifest['family_builds'][family]['driver_object_sha256']
        manifest_schema = json.loads((ROOT / 'docs/reference-manifest.schema.json').read_text())
        must_fail(f'{family} missing compiled object identity', lambda: jsonschema.validate(changed_manifest, manifest_schema))
    for path in ('../models/model.txt', '/tmp/model.txt', 'models/../model.txt'):
        must_fail('unsafe relative source/artifact path', lambda: safe_relative(path))


def helper_build_record(family, library_sha256):
    """Record production compile/link inputs and the resulting driver object."""
    work = OUT / family.replace('_', '-')
    build = OUT / 'build'
    source = ROOT / 'tools/oracle/extended_driver.cpp' if family == 'metric_edges' else work / 'driver.cpp'
    command = shlex.split(json.loads((build / 'compile_commands.json').read_text())[0]['command'])
    command[command.index('-o') + 1] = str(work / 'driver.o')
    command[command.index('-c') + 1] = str(source)
    if family == 'empty_queries':
        command.append(f'-I{ROOT / "tools/oracle"}')
    command.extend([f'-I{SOURCE}', f'-I{OUT}'])
    objects = sorted((build / 'CMakeFiles/lightgbm_objs.dir').rglob('*.o'))
    archive = build / 'external_libs/nanoarrow/libnanoarrow_static.a'
    link = [*command[:command.index('-o')], str(work / 'driver.o'), *map(str, objects), str(archive), '-o', str(work / 'driver')]
    return dict(compile_command=command, link_command=link, source_sha256=sha(source),
                shared_header_sha256=sha(ROOT / 'tools/oracle/driver_common.hpp'),
                compile_commands_sha256=sha(build / 'compile_commands.json'),
                driver_object_sha256=sha(work / 'driver.o'), loaded_library_sha256=library_sha256,
                object_sha256=[dict(path=str(p.relative_to(ROOT)), sha256=sha(p)) for p in [*objects, archive]])


def regenerate_empty_queries(fixtures):
    from generate_empty_queries import generate
    generate(fixtures)


def check_generated_executable(family, data):
    assert data['native_driver_sha256'] == sha(OUT / family.replace('_', '-') / 'driver'), f'{family} executable record mismatch'


def check_build_inputs(manifest):
    for family in ('metric_edges', 'empty_queries'):
        actual = helper_build_record(family, manifest['loaded_library']['loaded_sha256'])
        assert actual == manifest['family_builds'][family], f'{family} native build inputs changed'
        data = json.loads((ROOT / f'tests/fixtures/{family}.json').read_text())
        assert actual['source_sha256'] == data['native_driver_source_sha256']
        assert actual['compile_commands_sha256'] == manifest['build']['compile_commands_sha256']
        print(f'Source handle: tools/oracle/generate_{family}.py SHA-256 {data["generator_sha256"]}; native source {actual["source_sha256"]}')


def check_sort_headers(manifest):
    assert sort_header_record() == manifest['sort_headers'], 'native libc++ sort header identity changed'


def sort_header_record():
    command = shlex.split(json.loads((OUT / 'build/compile_commands.json').read_text())[0]['command'])
    directory = Path(command[command.index('-isysroot') + 1]) / 'usr/include/c++/v1/__algorithm'
    return dict(directory=str(directory), sha256={name: sha(directory / name) for name in SORT_HEADERS})
