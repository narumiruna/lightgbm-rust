"""Validate native evidence for the PR #1 review outcomes.

Expected numeric values come only from the native producers. Regeneration
compares every JSON field and model byte. There are no relink exclusions.
"""
import copy
import hashlib
import json
import math
import struct
from pathlib import Path

from common import COMMIT, ROOT, save_json, sha
from generate_predictions import BASE
from generate_review_refit import NAMES as REFIT_NAMES, specifications as refit_specs
from generate_review_cegb import NAMES as CEGB_NAMES, BASE as CEGB_BASE, specifications as cegb_specs
from generate_review_dart import NAMES as DART_NAMES, specifications as dart_specs
from generate_review_near_zero import NAMES as NEAR_ZERO_NAMES, BASE as NEAR_ZERO_BASE

NAMES = dict(review_refit=REFIT_NAMES, review_cegb=CEGB_NAMES, review_dart=DART_NAMES,
             review_near_zero=NEAR_ZERO_NAMES)


def shape_schema(value):
    """Strict structural schema; values remain native regeneration evidence."""
    if value is None:
        return dict(type='null')
    if isinstance(value, bool):
        return dict(type='boolean')
    if isinstance(value, (int, float)):
        return dict(type='number')
    if isinstance(value, str):
        return dict(type='string')
    if isinstance(value, list):
        variants = {json.dumps(shape_schema(item), sort_keys=True) for item in value}
        items = [json.loads(item) for item in sorted(variants)]
        return dict(type='array', minItems=len(value), maxItems=len(value),
                    items=items[0] if len(items) == 1 else dict(anyOf=items) if items else {})
    assert isinstance(value, dict)
    properties = {key: shape_schema(item) for key, item in value.items()}
    if 'name' in value:
        properties['name'] = dict(const=value['name'])
    if 'model_file' in value:
        pattern = r'^models/review_[a-z]+/[a-z0-9_]+\.txt$'
        if value['model_file'].startswith('models/review_near_zero/'):
            pattern = r'^models/review_near_zero/[a-z0-9_]+\.txt$'
        properties['model_file'] = dict(type='string', pattern=pattern)
    return dict(type='object', properties=properties, required=list(properties), additionalProperties=False)


def generate_schemas(fixtures=ROOT / 'tests/fixtures'):
    # Use the retained structure, not regenerated expectations. A source change
    # cannot silently rewrite a schema during a temporary reference check.
    for family in NAMES:
        data = json.loads((ROOT / f'tests/fixtures/{family}.json').read_text())
        schema = shape_schema(data)
        schema['$schema'] = 'https://json-schema.org/draft/2020-12/schema'
        schema['properties']['schema_version'] = dict(const=1)
        schema['properties']['upstream_commit'] = dict(const=COMMIT)
        save_json(Path(fixtures) / f'{family}.schema.json', schema)


def model_path(relative, family, fixtures):
    path = Path(relative)
    assert not path.is_absolute() and '..' not in path.parts
    assert str(path) == relative and path.parts[:2] == ('models', family)
    full = Path(fixtures) / path
    assert full.resolve().is_relative_to(Path(fixtures).resolve()) and full.is_file()
    return full


def matrix(value, rows, width):
    assert len(value) == rows and all(len(row) == width for row in value)


def snapshots(value, rows, classes, family, fixtures):
    if isinstance(value, dict):
        if {'raw', 'transformed', 'leaves'} <= value.keys():
            matrix(value['raw'], rows, classes)
            matrix(value['transformed'], rows, classes)
            matrix(value['leaves'], rows, value['num_trees'])
            assert value['current_iteration'] * classes == value['num_trees']
        if {'text', 'sha256'} <= value.keys():
            assert hashlib.sha256(value['text'].encode()).hexdigest() == value['sha256']
        if 'model_file' in value:
            full = model_path(value['model_file'], family, fixtures)
            digest = value.get('model_sha256', value.get('sha256'))
            if digest is not None:
                assert sha(full) == digest, 'native review model checksum changed'
            if 'num_trees' in value:
                assert sum(line.startswith('Tree=') for line in full.read_text().splitlines()) == value['num_trees']
        for item in value.values():
            snapshots(item, rows, classes, family, fixtures)
    elif isinstance(value, list):
        for item in value:
            snapshots(item, rows, classes, family, fixtures)


def identity(family, data, manifest):
    record = data['provenance'] if family == 'review_dart' else data
    assert record['source'] == manifest['source']
    assert record['compiler'] == manifest['build']['compiler']
    assert record['compile_commands_sha256'] == manifest['build']['compile_commands_sha256']
    assert record['library'] == {k: v for k, v in manifest['loaded_library'].items() if k != 'linked_libraries'}
    assert record['producer_sha256'] == sha(ROOT / f'tools/oracle/generate_{family}.py')
    hashes = {item['path']: item['sha256'] for item in manifest['reference_sources']}
    for field in ('source_sha256', 'native_sources'):
        if field in record:
            assert record[field] and all(hashes[path] == digest for path, digest in record[field].items())
    if family == 'review_dart':
        assert record['python_sources'] == {'basic.py': hashes['python-package/lightgbm/basic.py']}


def validate(family, data, manifest, fixtures=ROOT / 'tests/fixtures'):
    identity(family, data, manifest)
    cases = {case['name']: case for case in data['cases']}
    assert len(cases) == len(data['cases']) and set(cases) == NAMES[family]
    for case in cases.values():
        rows = case['rows']
        count, features = len(rows), len(rows[0])
        assert count and features and all(len(row) == features for row in rows)
        classes = case['params'].get('num_class', 1)
        snapshots(case, count, classes, family, fixtures)
    if family == 'review_refit':
        for name, extra in refit_specs():
            case = cases[name]
            assert case['params'] == dict(BASE, min_sum_hessian_in_leaf=0.0, **extra)
            training = case['training']
            rows = [[float(i - 12), float((i * 5) % 11 - 5)] for i in range(24)]
            labels = [i % 3 for i in range(24)] if extra['num_class'] == 3 else [int(a > 0) for a, _ in rows]
            assert training == dict(rows=rows, labels=labels, weights=[1., 2., .5, 1.] * 6, num_boost_round=4)
            assert case['rows'] == [[a + (1. if i % 3 == 0 else 0.), b] for i, (a, b) in enumerate(rows)]
            assert case['labels'] == labels[::-1] and case['weights'] == training['weights']
            assert case['decay_rate'] == .5 and case['distinct_sigmoid'] == extra['sigmoid'] * 1.00002
            token = next(t.split(':', 1)[1] for t in case['objective'].split() if t.startswith('sigmoid:'))
            assert float(token) == case['serialized_sigmoid'] != extra['sigmoid']
            assert case['native_refit']['fresh'] == case['native_refit']['reloaded']
            assert case['live_initial']['raw'] == case['initial_model']['raw']
            assert case['live_initial']['transformed'] != case['initial_model']['transformed']
            assert case['rounded_refit']['raw'] != case['native_refit']['fresh']['raw']
            matrix(case['leaf_pred'], 24, 4 * extra['num_class'])
    elif family == 'review_cegb':
        rows = [[g, b, c] for g in range(2) for b in range(2) for c in range(2) for _ in range(2)]
        labels = [20 * g + (4 if g == 0 else 2) * b + 2.5 * c * g for g, b, c in rows]
        for name, extra in cegb_specs():
            case = cases[name]
            assert case['params'] == {**CEGB_BASE, **extra}
            rounds = extra.get('num_iterations', 1)
            assert case['rows'] == rows and case['training']['rows'] == rows
            assert case['training']['labels'] == labels and case['training']['num_boost_round'] == rounds
            assert len(case['training']['expected_structure']) == len(case['native_split_gains']) == rounds
            matrix(case['raw'], 16, 1)
            matrix(case['leaves'], 16, rounds)
        root = cases['mixed_pending']['training']['expected_structure'][0]
        assert root['split_feature'] == 0
        assert root['left_child']['split_feature'] == root['right_child']['split_feature'] == 1
        assert root['right_child']['split_gain'] == 13.0
    elif family == 'review_dart':
        expected = {case['name']: case for case in dart_specs()['cases']}
        for name, inputs in expected.items():
            case = cases[name]
            for key, value in inputs.items():
                if key != 'initial_model':
                    assert case[key] == value, (name, key)
            assert (case['initial_model'] is not None) == inputs['initial_model']
            assert case['native_empty_bag_executed'] is False
            assert case['native_steps']
            assert case['native_steps'][0]['finished'] == (name == 'review_dart_initial_constant')
            for step in case['native_steps']:
                matrix(step['raw'], 4, case['params'].get('num_class', 1))
            if len(case['native_steps']) == 2 and case['params']['skip_drop'] == 0:
                assert case['native_steps'][1]['finished'] is True
                assert any(value < 0 for value in case['native_steps'][1]['shrinkage'])
                assert case['native_final_model'] is None and case['native_failed_model']
        assert 'NOT the native failed-round model' in data['checked_departure']
    elif family == 'review_near_zero':
        case = cases['negative_inclusive']
        z = struct.unpack('f', struct.pack('f', 1e-35))[0]
        training, validation = case['training'], case['validation']
        assert case['params'] == NEAR_ZERO_BASE and case['zero_threshold'] == z
        assert training['rows'] == case['rows'] == [[-1.], [-z], [0.], [1.]]
        assert training['labels'] == [0., 0., 1., 1.] and training['num_boost_round'] == 1
        assert validation['rows'] == [[v] for v in (-1., math.nextafter(-z, -math.inf), -z,
            math.nextafter(-z, 0.), 0., math.nextafter(z, 0.), z, math.nextafter(z, math.inf), 1.)]
        assert validation['labels'] == [0., 0., 0., 0., 1., 1., 1., 1., 1.]
        assert case['model_file'] == 'models/review_near_zero/negative_inclusive.txt'
        assert len(case['native_structure']) == 1 and case['native_structure'][0]['threshold'] == -z
        for name, dataset in [('training', training), ('validation', validation)]:
            count = len(dataset['rows'])
            assert len(dataset['scores']) == len(dataset['labels']) == count
            assert set(dataset['metrics']) == {'l2'} and len(dataset['metrics']['l2']) == 1
            matrix(case['dense'][name]['raw'], count, 1)
            matrix(case['dense'][name]['leaves'], count, 1)
        # Native Dataset bins raw -z; the dense prediction entry point clips it.
        assert training['scores'][1] != case['dense']['training']['raw'][1][0]
        assert validation['scores'][2] != case['dense']['validation']['raw'][2][0]
        assert 'keep_training_booster=True' in data['execution']
    else:
        raise AssertionError(f'unknown review family {family}')


def self_test(must_fail, read_json, manifest):
    for family in NAMES:
        original = read_json(ROOT / f'tests/fixtures/{family}.json')
        changed = copy.deepcopy(original)
        changed['cases'].pop()
        must_fail(f'{family} missing review case', lambda: validate(family, changed, manifest))
        changed = copy.deepcopy(original)
        if family == 'review_dart':
            changed['cases'][0]['native_steps'][0]['raw'][0][0] += .01
        elif family == 'review_refit':
            changed['cases'][0]['native_refit']['fresh']['raw'][0][0] += .01
        elif family == 'review_near_zero':
            changed['cases'][0]['dense']['training']['raw'][0][0] += .01
        else:
            changed['cases'][0]['raw'][0][0] += .01
        def same_native_values():
            assert changed == original, 'native review evidence changed'
        must_fail(f'{family} changed native value', same_native_values)
        changed = copy.deepcopy(original)
        changed['cases'][0]['rows'][0].pop()
        must_fail(f'{family} malformed review row', lambda: validate(family, changed, manifest))
        changed = copy.deepcopy(original)
        record = changed['provenance'] if family == 'review_dart' else changed
        record['producer_sha256'] = '0' * 64
        must_fail(f'{family} producer identity', lambda: identity(family, changed, manifest))
        changed = copy.deepcopy(original)
        record = changed['provenance'] if family == 'review_dart' else changed
        record['library']['loaded_sha256'] = '0' * 64
        must_fail(f'{family} loaded native image', lambda: identity(family, changed, manifest))
        changed = copy.deepcopy(original)
        changed['cases'][0]['last_valid_model' if family == 'review_dart' else 'initial_model' if family == 'review_refit' else 'model_file'] = (
            '../model.txt' if family in ('review_cegb', 'review_near_zero') else {'model_file': '../model.txt'})
        must_fail(f'{family} unsafe model path', lambda: validate(family, changed, manifest))
    changed = read_json(ROOT / 'tests/fixtures/review_cegb.json')
    next(c for c in changed['cases'] if c['name'] == 'mixed_pending')['training']['expected_structure'][0]['right_child']['split_feature'] = 2
    must_fail('CEGB native pending-leaf choice', lambda: validate('review_cegb', changed, manifest))
    changed = read_json(ROOT / 'tests/fixtures/review_refit.json')
    changed['cases'][0]['params']['sigmoid'] = changed['cases'][0]['serialized_sigmoid']
    must_fail('refit full-precision parameter', lambda: validate('review_refit', changed, manifest))
    changed = read_json(ROOT / 'tests/fixtures/review_dart.json')
    changed['cases'][0]['native_empty_bag_executed'] = True
    must_fail('DART excluded empty native learner', lambda: validate('review_dart', changed, manifest))
    changed = read_json(ROOT / 'tests/fixtures/review_near_zero.json')
    changed['cases'][0]['training']['scores'][1] = changed['cases'][0]['dense']['training']['raw'][1][0]
    must_fail('native near-zero buffer/dense distinction', lambda: validate('review_near_zero', changed, manifest))


if __name__ == '__main__':
    generate_schemas()
