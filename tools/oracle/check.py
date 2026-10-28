"""Validate retained evidence; optionally regenerate it without changing saved files."""
import argparse
import copy
import ctypes
import json
import math
import re
from pathlib import Path
import tempfile
from types import SimpleNamespace

import jsonschema

from common import COMMIT, OUT, ROOT, SOURCE, load_reference, sha, verify_loaded_library
from manifest import artifact_paths
from check_workflows import validate as phase8_shapes
from check_final import NAMES as FINAL_FAMILIES, validate as final_shapes, exact_payload as final_compare
from check_review import NAMES as REVIEW_FAMILIES, validate as review_shapes

PREDICTION_CASES = {"numeric_l2", "binary", "numeric_weighted", "missing_nan", "missing_zero", "categorical",
                    "multiclass", "single_leaf", "regression_sqrt", "binary_sigmoid", "multiclassova", "poisson",
                    "gamma", "tweedie", "cross_entropy", "cross_entropy_lambda", "rf_average", "missing_none"}
REQUIRED = {
    "prediction": PREDICTION_CASES,
    "prediction_regressions": {"missing_only", "missing_direction_tie"},
    "prediction_ranges": {"multiclass_empty_class", "multiclass_start1_count1", "multiclass_start1_count2",
                          "rf_average_start1_count1", "rf_average_start1_count2"},
    "rng": {"seed0", "seed1", "seed42"},
    "objectives": {"l2", "l2_weighted", "binary", "binary_weighted"},
    "histograms": {"numeric_split", "numeric_split_l2"},
    "bins": {"numeric", "nan_missing", "zero_missing", "constant", "max_bin4", "zero_separated", "missing_disabled",
             "max_bin2", "max_bin3", "min_data_in_bin3", "adjacent_doubles", "categorical_ties", "categorical_missing", "categorical_rare"},
}
from generate_extended import specifications, OBJECTIVES, METRICS
EXTENDED = specifications()
REQUIRED.update({family: {c['name'] for c in cases} for family, cases in EXTENDED.items()})
REQUIRED['prediction_extended'] = {o + '_extended' for o in OBJECTIVES} | {
    'multiclass_absent_extended', 'multiclassova_absent_extended',
    'categorical_onehot', 'categorical_onehot_weighted', 'categorical_partition', 'categorical_partition_weighted'}
REQUIRED['prediction_sparse'] = {name + '_csr' for name in PREDICTION_CASES}
REQUIRED['prediction_linear'] = {f'linear_{kind}_lambda{level}' for kind in ('independent', 'rank_deficient') for level in ('0', '05')}
from generate_training_controls import specifications as control_specifications
from generate_feature_info import specifications as feature_info_specifications
from generate_linear_training import NAMES as LINEAR_TRAINING_NAMES
CONTROL_SPECS = control_specifications()
REQUIRED.update({family: {name for name, _ in specs} for family, specs in CONTROL_SPECS.items()})
REQUIRED['prediction_linear_training'] = LINEAR_TRAINING_NAMES
REQUIRED['data_feature_info'] = {c['name'] for c in feature_info_specifications()}
REQUIRED['explain'] = {name + suffix for name in PREDICTION_CASES | REQUIRED['prediction_ranges'] for suffix in ('', '_csr')}
REQUIRED['sampling'] = (REQUIRED['prediction_boosting'] - {'rf_averaging', 'rf_binary'}) | {'feature_filtered'} | {
    'quantization_' + rounding + hessian for rounding in ('stochastic', 'deterministic') for hessian in ('', '_constant')}
NEW_FAMILIES = ['data_feature_info', 'explain', 'sampling', *CONTROL_SPECS, 'prediction_linear_training']
from generate_workflow import NAMES as WORKFLOW_NAMES
from generate_cv import NAMES as CV_NAMES
from generate_operations import NAMES as OPERATION_NAMES
PHASE8_FAMILIES = {'workflow': WORKFLOW_NAMES, 'cv': CV_NAMES, 'prediction_operations': OPERATION_NAMES}
REQUIRED.update(PHASE8_FAMILIES)
from generate_constraints_extra import NAMES as EXTRA_CONSTRAINT_NAMES, BASE as EXTRA_CONSTRAINT_BASE, specifications as extra_constraint_specifications
REQUIRED['prediction_constraints_extra'] = EXTRA_CONSTRAINT_NAMES
from check_bundling import NAMES as BUNDLING_NAMES, validate as bundling_shapes, stable_payload as bundling_payload
REQUIRED['bundling'] = BUNDLING_NAMES
REQUIRED.update(FINAL_FAMILIES)
REQUIRED.update(REVIEW_FAMILIES)


def schema_name(family):
    if family in PHASE8_FAMILIES or family in FINAL_FAMILIES or family in REVIEW_FAMILIES or family in ('prediction_constraints_extra', 'bundling'):
        return f'{family}.schema.json'
    return 'prediction.schema.json' if family.startswith('prediction') else f'{family}.schema.json' if family in EXTENDED or family in NEW_FAMILIES else 'primitives.schema.json'


def read_json(path):
    def reject(value):
        raise ValueError(f"non-finite JSON token {value}")
    value = json.loads(Path(path).read_text(), parse_constant=reject)
    def finite(item):
        if isinstance(item, float):
            assert math.isfinite(item), "non-finite JSON number"
        elif isinstance(item, list):
            for v in item:
                finite(v)
        elif isinstance(item, dict):
            for v in item.values():
                finite(v)
    finite(value)
    return value


def coverage(family, data):
    names = [c["name"] for c in data["cases"]]
    assert len(names) == len(set(names)), f"duplicate {family} case"
    assert set(names) == REQUIRED[family], f"missing or unexpected {family} cases: {set(names) ^ REQUIRED[family]}"


def prediction_shapes(data):
    for c in data["cases"]:
        rows = len(c["rows"])
        features = len(c["rows"][0])
        assert all(len(row) == features for row in c["rows"]), c["name"]
        classes = c["params"].get("num_class", 1)
        for key in ("raw", "transformed"):
            assert len(c[key]) == rows and all(len(r) == classes for r in c[key]), (c["name"], key)
        trees = sum(line.startswith("Tree=") for line in (ROOT / "tests/fixtures" / c["model_file"]).read_text().splitlines())
        selected_trees = trees
        if "start_iteration" in c["params"]:
            start = c["params"]["start_iteration"] * classes
            selected_trees = min(trees - start, c["params"]["num_iteration"] * classes)
        assert len(c["leaves"]) == rows and all(len(r) == selected_trees for r in c["leaves"]), c["name"]
        if 'sparse' in c:
            sparse = c['sparse']
            assert sparse['shape'] == [rows, features]
            assert len(sparse['indptr']) == rows + 1 and sparse['indptr'][0] == 0
            assert sparse['indptr'][-1] == len(sparse['indices']) == len(sparse['data'])
            for row in range(rows):
                start, end = sparse['indptr'][row:row + 2]
                indices = sparse['indices'][start:end]
                assert start <= end and indices == sorted(set(indices))
                rebuilt = [0.] * features
                for index, value in zip(indices, sparse['data'][start:end]):
                    assert 0 <= index < features
                    rebuilt[index] = value
                assert rebuilt == c['rows'][row]
        train = c.get("training")
        if train:
            assert len(train["rows"]) == len(train["labels"]), c["name"]
            assert len(train["expected_structure"]) == trees, c["name"]
            if "weights" in train:
                assert len(train["weights"]) == len(train["labels"]), c["name"]
            if 'groups' in train:
                assert sum(train['groups']) == len(train['labels']) and all(g > 0 for g in train['groups'])
            if 'positions' in train:
                assert len(train['positions']) == len(train['labels'])
            if 'steps' in train:
                assert len(train['steps']) == train['num_boost_round']
                for iteration, step in enumerate(train['steps']):
                    assert step['iteration'] == iteration
                    assert len(step['raw']) == rows and all(len(r) == classes for r in step['raw'])
                    assert len(step['tree_shrinkage']) == (iteration + 1) * classes
                assert all(abs(a - e) <= 1e-12 + 1e-10 * abs(e) for ar, er in zip(train['steps'][-1]['raw'], c['raw']) for a, e in zip(ar, er))


def extended_shapes(family, data):
    for c in data['cases']:
        n = len(c.get('labels', c.get('sample_values', [])))
        if 'weights' in c:
            assert len(c['weights']) == n
        if 'groups' in c:
            assert sum(c['groups']) == n and all(g > 0 for g in c['groups'])
        if 'positions' in c:
            assert len(c['positions']) == n
        if family in ('extended_objectives', 'metrics'):
            classes = c['params'].get('num_class', 1)
            assert len(c['scores']) == n * classes
            if 'source_class_need_train' in c:
                assert len(c['source_class_need_train']) == classes and c['model_objective']
            if family == 'extended_objectives':
                assert len(c['initial_scores']) == classes
                for step in [c, *c.get('steps', [])]:
                    assert len(step['scores']) == len(step['gradients']) == len(step['hessians']) == n * classes
                for renewal in c.get('renewals', []):
                    assert renewal['rows'] and all(0 <= row < n for row in renewal['rows'])
        elif family == 'dataset_metadata':
            assert len(c['expected']['labels']) == n
            if 'initial_scores' in c:
                assert len(c['initial_scores']) == n * c['num_class']
                assert len(c['expected']['initial_scores']) == len(c['initial_scores'])
            if 'binning' in c:
                b, expected = c['binning'], c['expected']['binning']
                features = len(b['max_bin_by_feature'])
                assert len(b['forced_upper_bounds']) == len(expected['upper_bounds']) == features
                for inputs, outputs in [('rows', 'training_bins'), ('validation_rows', 'validation_bins')]:
                    assert len(b[inputs]) == len(expected[outputs])
                    for row, bins in zip(b[inputs], expected[outputs]):
                        assert len(row) == len(bins) == features
                        assert all(0 <= value < len(expected['upper_bounds'][f]) for f, value in enumerate(bins))
        elif family == 'forced_bins':
            assert len(c['probe_values']) == len(c['value_bins'])
            assert len(c['upper_bounds']) == c['num_bin']
            assert all(0 <= b < c['num_bin'] for b in c['value_bins'])


def new_shapes(family, data):
    if family == 'data_feature_info':
        for c in data['cases']:
            assert c['rows'] and all(v in ('-Infinity', 'Infinity') for v in c['rows'] if isinstance(v, str))
    elif family == 'explain':
        assert {c['name'] for c in data['expected_errors']} == REQUIRED['prediction_linear']
        assert all('not implemented for linear trees' in c['expected_error'] for c in data['expected_errors'])
        for c in data['cases']:
            n, features = len(c['rows']), len(c['rows'][0])
            classes = c['params'].get('num_class', 1)
            assert len(c['raw']) == len(c['contributions']) == n
            assert len(c['importance_split']) == len(c['importance_gain']) == features
            assert all(len(row) == features for row in c['rows'])
            for raw, contributions in zip(c['raw'], c['contributions']):
                assert len(raw) == classes and len(contributions) == classes * (features + 1)
                for k, expected in enumerate(raw):
                    total = sum(contributions[k * (features + 1):(k + 1) * (features + 1)])
                    assert abs(total - expected) <= 1e-12 + 1e-10 * abs(expected), c['name']
            if 'sparse' in c:
                sparse = c['sparse']
                assert sparse['shape'] == [n, features] and len(sparse['indptr']) == n + 1
                assert sparse['indptr'][0] == 0 and sparse['indptr'][-1] == len(sparse['indices']) == len(sparse['data'])
                for row in range(n):
                    start, end = sparse['indptr'][row:row + 2]
                    assert 0 <= start <= end <= len(sparse['data'])
                    indices = sparse['indices'][start:end]
                    assert indices == sorted(set(indices)) and all(0 <= f < features for f in indices)
    elif family == 'sampling':
        for c in data['cases']:
            n, features = len(c['rows']), len(c['rows'][0])
            classes = c['params'].get('num_class', 1)
            assert len(c['labels']) == n
            if 'weights' in c:
                assert len(c['weights']) == n
            if 'positions' in c:
                assert len(c['positions']) == n
            assert len(c['steps']) == c['num_steps']
            for iteration, step in enumerate(c['steps']):
                assert step['iteration'] == iteration
                if c['kind'] == 'rows':
                    assert len(c['scores']) == n * classes
                    assert all(len(step[key]) == n * classes for key in ('input_gradients', 'input_hessians', 'gradients', 'hessians'))
                    selected = step['selected_rows']
                    assert len(selected) == len(set(selected)) and all(0 <= row < n for row in selected)
                    assert (not selected and step['selected_count'] == n) if step['all_rows'] else len(selected) == step['selected_count']
                    if c['params'].get('data_sample_strategy') == 'goss':
                        assert step['all_rows'] == (iteration < int(1. / c['params']['learning_rate']))
                    if c['params'].get('bagging_by_query'):
                        boundaries = [0]
                        for group in c['groups']:
                            boundaries.append(boundaries[-1] + group)
                        assert boundaries[-1] == n
                        assert selected == [row for q in step['selected_queries'] for row in range(boundaries[q], boundaries[q + 1])]
                elif c['kind'] == 'features':
                    eligible = c['eligible_features']
                    assert eligible == sorted(set(eligible)) and all(0 <= f < features for f in eligible)
                    assert len(c['inner_feature_indices']) == features
                    assert sorted(c['inner_feature_indices'][f] for f in eligible) == list(range(len(eligible)))
                    assert all(c['inner_feature_indices'][f] == -1 for f in range(features) if f not in eligible)
                    assert len(step['by_tree']) == len(eligible) and len(step['by_node']) == 2
                    assert all(len(mask) == len(eligible) for mask in step['by_node'])
                elif c['kind'] == 'dart':
                    assert all(0 <= tree < iteration for tree in step['drop_indices'])
                    if not c['params'].get('uniform_drop'):
                        assert len(step['tree_weights']) == iteration + 1
                        if not c['params'].get('xgboost_dart_mode'):
                            assert abs(sum(step['tree_weights']) - step['sum_weight']) <= 1e-12
                elif c['kind'] == 'quantization':
                    assert len(c['gradients']) == len(c['hessians']) == n
                    assert len(step['packed_hessian_gradient']) == n * 2
                    assert step['grad_scale'] > 0 and step['hess_scale'] > 0


def extra_constraint_shapes(data, manifest=None):
    assert data['producer_sha256'] == sha(ROOT / 'tools/oracle/generate_constraints_extra.py')
    if manifest:
        assert data['library'] == {k: v for k, v in manifest['loaded_library'].items() if k != 'linked_libraries'}
        assert data['compiler'] == manifest['build']['compiler']
        assert data['compile_commands_sha256'] == manifest['build']['compile_commands_sha256']
    expected = dict(extra_constraint_specifications())
    for c in data['cases']:
        assert c['params'] == {**EXTRA_CONSTRAINT_BASE, **expected[c['name']]}
        train, rows = c['training'], c['rows']
        n, features = len(rows), len(rows[0])
        assert train['rows'] == rows
        assert len(train['labels']) == len(train['weights']) == n
        assert all(len(row) == features for row in rows)
        assert len(c['raw']) == len(c['leaves']) == n
        assert all(len(row) == 1 for row in c['raw'])
        path = ROOT / 'tests/fixtures' / c['model_file']
        assert path.resolve().is_relative_to(ROOT / 'tests/fixtures/models/constraints_extra')
        trees = sum(line.startswith('Tree=') for line in path.read_text().splitlines())
        assert trees == train['num_boost_round'] == 4
        assert all(len(row) == trees for row in c['leaves'])


def parameter_inventory(data):
    assert data['schema_version'] == 1 and data['upstream_commit'] == COMMIT
    aliases, defaults, types = data['aliases'], data['declared_defaults'], data['types']
    assert len(aliases) == 141 and set(aliases) == set(defaults)
    assert set(aliases) - set(types) == {'task'} and set(types) <= set(aliases)
    names = [alias for values in aliases.values() for alias in values]
    assert len(names) == len(set(names)) == 167 and not set(names) & set(aliases)
    assert types['categorical_feature'] == types['ignore_column'] == 'vector<int>'
    assert types['interaction_constraints'] == 'vector<vector<int>>'
    for name, kind in types.items():
        value = defaults[name]
        if name in ('categorical_feature', 'ignore_column', 'interaction_constraints'):
            assert value == ''  # These Config fields hold source text, not parsed vectors.
        elif kind.startswith('vector<'):
            assert isinstance(value, list)
        elif kind == 'bool':
            assert isinstance(value, bool)
        elif kind == 'int':
            assert type(value) is int
        elif kind == 'double':
            assert type(value) in (int, float)
        else:
            assert kind == 'string' and isinstance(value, str)


def source_registry_coverage():
    inventory = read_json(ROOT / 'tests/fixtures/parameters.json')
    config = (SOURCE / 'src/io/config_auto.cpp').read_text()
    aliases = config.split('Config::parameter2aliases()', 1)[1].split('return map;', 1)[0]
    assert {name: re.findall(r'"([^"]+)"', values) for name, values in
            re.findall(r'\{"([^"]+)", \{([^}]*)\}\}', aliases)} == inventory['aliases']
    types = config.split('Config::ParameterTypes()', 1)[1].split('return map;', 1)[0]
    assert dict(re.findall(r'\{"([^"]+)", "([^"]+)"\}', types)) == inventory['types']
    for source, function, required in [('objective/objective_function.cpp', 'ObjectiveFunction::CreateObjectiveFunction(const std::string& type', OBJECTIVES),
                                        ('metric/metric.cpp', 'Metric::CreateMetric(', METRICS)]:
        cpu = (SOURCE / 'src' / source).read_text().split(function, 1)[1].split('#endif  // USE_CUDA', 1)[1].split('#ifdef USE_CUDA', 1)[0]
        names = set(re.findall(r'type == std::string\("([^"]+)"\)', cpu)) - {'custom'}
        assert names == set(required), f'CPU factory coverage incomplete: {source}'
    print('Pinned source registry audit passed: 141 canonical parameters, 167 aliases, 16 CPU objectives, 24 CPU metrics')


def compare_exact(actual, expected):
    assert actual == expected, 'native evidence changed'


def helper_identities(manifest):
    for name, record in manifest['native_helpers'].items():
        assert record['loaded_library_sha256'] == manifest['loaded_library']['loaded_sha256']
        assert record['source_sha256'] == sha(ROOT / 'tools/oracle' / f'{name}.cpp')


def artifact_checksum(entry):
    path = ROOT / entry['path']
    from check_final import safe_relative
    safe_relative(entry['path'])
    assert path.resolve().is_relative_to(ROOT), 'artifact path escapes repository'
    assert sha(path) == entry['sha256'], f'artifact SHA-256 mismatch: {path}'


def validate_saved():
    fixture_dir = ROOT / "tests/fixtures"
    manifest = read_json(ROOT / "docs/reference-manifest.json")
    jsonschema.Draft202012Validator(read_json(ROOT / "docs/reference-manifest.schema.json")).validate(manifest)
    assert manifest["loaded_library"]["path"] == manifest["loaded_library"]["loaded_path"]
    assert manifest["loaded_library"]["sha256"] == manifest["loaded_library"]["loaded_sha256"]
    assert manifest['extended_driver']['loaded_library_sha256'] == manifest['loaded_library']['loaded_sha256']
    for key, name in [('driver', 'primitive_driver.cpp'), ('extended_driver', 'extended_driver.cpp')]:
        assert manifest[key]['source_sha256'] == sha(ROOT / 'tools/oracle' / name)
        assert manifest[key]['shared_header_sha256'] == sha(ROOT / 'tools/oracle/driver_common.hpp')
    helper_identities(manifest)
    for entry in manifest['reference_sources']:
        from check_final import safe_relative
        safe_relative(entry['path'])
    paths = [a["path"] for a in manifest["artifacts"]]
    assert len(paths) == len(set(paths)), "duplicate artifact"
    assert set(paths) == {str(p.relative_to(ROOT)) for p in artifact_paths()}, "artifact inventory incomplete"
    for entry in manifest["artifacts"]:
        artifact_checksum(entry)
    for family in REQUIRED:
        data = read_json(fixture_dir / f"{family}.json")
        schema = schema_name(family)
        jsonschema.Draft202012Validator(read_json(fixture_dir / schema)).validate(data)
        coverage(family, data)
        if family in REVIEW_FAMILIES:
            review_shapes(family, data, manifest)
        elif family in FINAL_FAMILIES:
            final_shapes(family, data, manifest)
        elif family in PHASE8_FAMILIES:
            phase8_shapes(family, data, manifest)
        elif family == 'prediction_constraints_extra':
            extra_constraint_shapes(data, manifest)
        elif family == 'bundling':
            bundling_shapes(data, manifest)
        elif family.startswith("prediction"):
            prediction_shapes(data)
        elif family in EXTENDED:
            extended_shapes(family, data)
        else:
            new_shapes(family, data)
        if family in CONTROL_SPECS:
            expected = dict(CONTROL_SPECS[family])
            for c in data['cases']:
                assert all(c['params'].get(key) == value for key, value in expected[c['name']].items()), c['name']
    assert {c['objective'] for c in read_json(fixture_dir / 'extended_objectives.json')['cases']} == set(OBJECTIVES)
    assert {c['metric'] for c in read_json(fixture_dir / 'metrics.json')['cases']} == set(METRICS)
    parameter_inventory(read_json(fixture_dir / 'parameters.json'))
    contract = read_json(fixture_dir / "benchmark-contract.json")
    assert contract['performance_budgets'] == dict(training_ratio=1.5, inference_ratio=1.25, peak_memory_ratio=1.5)
    from benchmark import contract as executable_contract
    assert contract == executable_contract(), "benchmark definition changed"
    baseline = read_json(fixture_dir / "reference-baseline.json")
    assert {c["name"] for c in baseline["cases"]} == {c["name"] for c in contract["cases"]}
    for c in baseline["cases"]:
        assert len(c["training_seconds"]) >= 5 and len(c["inference_seconds"]) >= 5
        bound = next(b for b in contract["cases"] if b["name"] == c["name"])
        assert all(v <= bound["quality_bound"] if bound["comparison"] == "<=" else v >= bound["quality_bound"] for v in c["quality"])
    print(f"Saved schemas, coverage, shapes, quality gates, and {len(paths)} artifact hashes passed")
    return manifest


def must_fail(label, operation):
    try:
        operation()
    except (AssertionError, ValueError, jsonschema.ValidationError):
        print(f"Expected rejection passed: {label}")
        return
    raise AssertionError(f"mutation was accepted: {label}")


def self_test(reference):
    from check_final import self_test as final_self_test
    final_self_test(must_fail)
    saved = read_json(ROOT / "tests/fixtures/prediction.json")
    missing = copy.deepcopy(saved)
    missing["cases"].pop()
    must_fail("missing prediction case", lambda: coverage("prediction", missing))
    shape = copy.deepcopy(saved)
    shape["cases"][0]["raw"].pop()
    must_fail("prediction row shape", lambda: prediction_shapes(shape))
    changed = copy.deepcopy(saved)
    changed["cases"][0]["raw"][0][0] += 0.01
    must_fail("prediction value", lambda: compare_prediction_values(changed, saved))
    manifest = read_json(ROOT / "docs/reference-manifest.json")
    from check_review import self_test as review_self_test
    review_self_test(must_fail, read_json, manifest)
    changed_hash = dict(manifest['artifacts'][0], sha256='0' * 64)
    must_fail('changed artifact checksum', lambda: artifact_checksum(changed_hash))
    del manifest["loaded_library"]
    schema = jsonschema.Draft202012Validator(read_json(ROOT / "docs/reference-manifest.schema.json"))
    must_fail("manifest without loaded library", lambda: schema.validate(manifest))
    for family in [*EXTENDED, 'prediction_extended', 'prediction_sparse', 'prediction_linear', *NEW_FAMILIES, 'prediction_constraints_extra']:
        original = read_json(ROOT / f'tests/fixtures/{family}.json')
        missing = copy.deepcopy(original)
        missing['cases'].pop()
        must_fail(f'{family} missing case', lambda: coverage(family, missing))
        changed = copy.deepcopy(original)
        c = changed['cases'][0]
        if family == 'extended_objectives':
            c['gradients'][0] += 0.01
        elif family == 'metrics':
            c['results'][0]['value'] += 0.01
        elif family == 'dataset_metadata':
            c['expected']['labels'][0] += 0.01
        elif family == 'forced_bins':
            c['value_bins'][0] += 1
        elif family == 'data_feature_info':
            c['expected'] += 'changed'
        elif family == 'sampling':
            c['steps'][0]['selected_rows'][0] += 1
        else:
            c['raw'][0][0] += 0.01
        must_fail(f'{family} changed value', lambda: compare_exact(changed, original))
        malformed = copy.deepcopy(original)
        if family == 'extended_objectives':
            malformed['cases'][0]['hessians'].pop()
        elif family == 'metrics':
            malformed['cases'][0]['results'] = []
        elif family == 'dataset_metadata':
            malformed['cases'][0]['expected']['labels'].pop()
        elif family == 'forced_bins':
            malformed['cases'][0]['upper_bounds'].pop()
        elif family == 'data_feature_info':
            malformed['cases'][0]['rows'] = []
        elif family == 'explain':
            malformed['cases'][0]['contributions'][0].pop()
        elif family == 'sampling':
            malformed['cases'][0]['steps'][0]['gradients'].pop()
        else:
            malformed['cases'][0]['raw'].pop()
        def reject_shape():
            jsonschema.Draft202012Validator(read_json(ROOT / 'tests/fixtures' / schema_name(family))).validate(malformed)
            if family == 'prediction_constraints_extra':
                extra_constraint_shapes(malformed)
            elif family.startswith('prediction'):
                prediction_shapes(malformed)
            elif family in EXTENDED:
                extended_shapes(family, malformed)
            else:
                new_shapes(family, malformed)
        must_fail(f'{family} malformed shape', reject_shape)
    for family in PHASE8_FAMILIES:
        original = read_json(ROOT / f'tests/fixtures/{family}.json')
        missing = copy.deepcopy(original)
        missing['cases'].pop()
        must_fail(f'{family} missing case', lambda: coverage(family, missing))
        changed = copy.deepcopy(original)
        changed['cases'][0]['labels'][0] += 1
        must_fail(f'{family} changed native evidence', lambda: compare_exact(changed, original))
        malformed = copy.deepcopy(original)
        malformed['cases'][0]['labels'].pop()
        must_fail(f'{family} malformed shape', lambda: phase8_shapes(family, malformed))
        malformed = copy.deepcopy(original)
        del malformed['provenance']['loaded_library_sha256']
        must_fail(f'{family} missing native provenance', lambda: jsonschema.Draft202012Validator(
            read_json(ROOT / 'tests/fixtures' / schema_name(family))).validate(malformed))
    malformed = read_json(ROOT / 'tests/fixtures/workflow.json')
    malformed['cases'][0]['events'][0]['callback'] = 'before_last'
    must_fail('workflow callback order mutation', lambda: phase8_shapes('workflow', malformed))
    malformed = read_json(ROOT / 'tests/fixtures/cv.json')
    malformed['cases'][0]['steps'][0]['aggregate'][0]['metric_std_dev'] += 0.01
    must_fail('cv population deviation mutation', lambda: phase8_shapes('cv', malformed))
    malformed = read_json(ROOT / 'tests/fixtures/prediction_operations.json')
    refit = next(c for c in malformed['cases'] if c['operation'] == 'refit')
    refit['refit_steps'][0]['gradients'].pop()
    must_fail('refit gradient shape mutation', lambda: phase8_shapes('prediction_operations', malformed))
    malformed = read_json(ROOT / 'tests/fixtures/prediction_operations.json')
    extreme = next(c for c in malformed['cases'] if c['operation'] == 'extreme_gain_diagnostic')
    extreme['importance_gain'][0] = 'not-a-number'
    must_fail('exceptional gain encoding mutation', lambda: jsonschema.Draft202012Validator(
        read_json(ROOT / 'tests/fixtures/prediction_operations.schema.json')).validate(malformed))
    original = read_json(ROOT / 'tests/fixtures/bundling.json')
    malformed = copy.deepcopy(original)
    malformed['cases'].pop()
    must_fail('bundling missing case', lambda: coverage('bundling', malformed))
    malformed = copy.deepcopy(original)
    malformed['cases'][0]['raw_predictions'][0] += .01
    must_fail('bundling changed value', lambda: compare_exact(bundling_payload(malformed), bundling_payload(original)))
    malformed = copy.deepcopy(original)
    malformed['cases'][0]['bins'][0].pop()
    must_fail('bundling malformed shape', lambda: bundling_shapes(malformed))
    malformed = copy.deepcopy(original)
    malformed['cases'][0]['groups'][0]['decoded_conflicts'] += 1
    must_fail('bundling conflict mutation', lambda: bundling_shapes(malformed))
    malformed = copy.deepcopy(original)
    malformed['cases'][0]['model_sha256'] = '0' * 64
    must_fail('bundling model checksum mutation', lambda: bundling_shapes(malformed))
    for helper in ('data_feature_info', 'sampling_driver', 'refit_driver', 'bundling_driver'):
        malformed = read_json(ROOT / 'docs/reference-manifest.json')
        del malformed['native_helpers'][helper]['source_sha256']
        must_fail(f'{helper} missing identity', lambda: schema.validate(malformed))
        for field in ('source_sha256', 'loaded_library_sha256'):
            changed = read_json(ROOT / 'docs/reference-manifest.json')
            changed['native_helpers'][helper][field] = '0' * 64
            must_fail(f'{helper} changed {field}', lambda: helper_identities(changed))
    original = read_json(ROOT / 'tests/fixtures/parameters.json')
    malformed = copy.deepcopy(original)
    malformed['aliases']['boosting'].pop()
    must_fail('missing frozen parameter alias', lambda: parameter_inventory(malformed))
    malformed = copy.deepcopy(original)
    malformed['declared_defaults']['num_threads'] = '0'
    must_fail('parameter default type mutation', lambda: parameter_inventory(malformed))
    malformed = read_json(ROOT / 'tests/fixtures/prediction_constraints_extra.json')
    malformed['cases'][0]['model_file'] = '../constraints_models/deep_monotone_basic.txt'
    must_fail('extra constraints path escape', lambda: jsonschema.Draft202012Validator(
        read_json(ROOT / 'tests/fixtures/prediction_constraints_extra.schema.json')).validate(malformed))
    malformed = read_json(ROOT / 'tests/fixtures/prediction_constraints_extra.json')
    malformed['producer_sha256'] = '0' * 64
    must_fail('extra constraints producer identity mutation', lambda: extra_constraint_shapes(malformed))
    malformed = read_json(ROOT / 'tests/fixtures/prediction_operations.json')
    refit = next(c for c in malformed['cases'] if c['name'] == 'refit_goss_querybag_decay05')
    refit['backend_params']['bagging_by_query'] = True
    must_fail('GOSS query bagging normalization mutation', lambda: phase8_shapes('prediction_operations', malformed))
    if reference:
        _, record = load_reference()
        from lightgbm.libpath import _LIB
        wrong = copy.deepcopy(record)
        wrong["library"]["sha256"] = "0" * 64
        must_fail("changed loaded-library hash", lambda: verify_loaded_library(_LIB, wrong))
        fake = SimpleNamespace(_name=str(ROOT / "unrelated-wheel/lib_lightgbm.dylib"), LGBM_GetLastError=_LIB.LGBM_GetLastError)
        must_fail("unrelated wheel path", lambda: verify_loaded_library(fake, record))
        fake._name = record["library"]["path"]
        fake.LGBM_GetLastError = ctypes.CDLL(None).printf
        must_fail("unrelated loaded image despite claimed name", lambda: verify_loaded_library(fake, record))


def compare_prediction_values(actual, expected):
    coverage("prediction", actual)
    for a, e in zip(actual["cases"], expected["cases"]):
        assert a["name"] == e["name"]
        for kind in ("raw", "transformed"):
            assert len(a[kind]) == len(e[kind])
            for ar, er in zip(a[kind], e[kind]):
                assert len(ar) == len(er)
                assert all(abs(av - ev) <= 1e-12 + 1e-10 * abs(ev) for av, ev in zip(ar, er))
        assert a["leaves"] == e["leaves"]


def check_reference(manifest):
    _, record = load_reference()
    assert record["library"]["loaded_sha256"] == manifest["loaded_library"]["loaded_sha256"]
    assert record["library"]["loaded_path"] == manifest["loaded_library"]["loaded_path"]
    assert record["compiler"] == manifest["build"]["compiler"]
    assert record["source"] == manifest["source"]
    print(f'Pinned source handle: {SOURCE}@{COMMIT} version {manifest["source"]["version"]}')
    print(f'Loaded native image: {record["library"]["loaded_path"]} SHA-256 {record["library"]["loaded_sha256"]}')
    print(f'Source handle: tools/oracle/generate_workflow_extended.py SHA-256 {sha(ROOT / "tools/oracle/generate_workflow_extended.py")}')
    from generate_predictions import generate as predictions
    from generate_regressions import generate as regressions
    from generate_primitives import generate as primitives
    from generate_extended import generate as extended
    from generate_prediction_extended import generate as extended_predictions
    from generate_linear import generate as linear
    from generate_feature_info import generate as feature_info
    from generate_explain import generate as explain
    from generate_training_controls import generate as controls
    from generate_sampling import generate as sampling
    from generate_linear_training import generate as linear_training
    from generate_workflow import generate as workflow
    from generate_cv import generate as cv
    from generate_operations import generate as operations
    from generate_workflow_schemas import generate as workflow_schemas
    from generate_constraints_extra import generate as extra_constraints
    from check_bundling import regenerate as bundling, generate_schema as bundling_schema
    from check_final import check_build_inputs, check_sort_headers, check_generated_executable
    from generate_workflow_extended import generate as workflow_extended
    from generate_metric_edges import generate as metric_edges
    from check_final import regenerate_empty_queries as empty_queries
    from generate_final_schemas import generate as final_schemas
    from check_review import generate_schemas as review_schemas
    from generate_review_refit import generate as review_refit
    from generate_review_cegb import generate as review_cegb
    from generate_review_dart import generate as review_dart
    from generate_review_near_zero import generate as review_near_zero
    source_registry_coverage()
    check_sort_headers(manifest)
    check_build_inputs(manifest)
    with tempfile.TemporaryDirectory(prefix="oracle-check-", dir=OUT) as directory:
        output = Path(directory)
        predictions(output)
        regressions(output)
        primitives(output)
        extended(output)
        extended_predictions(output)
        linear(output)
        feature_info(output)
        explain(output)
        controls(output)
        sampling(output)
        linear_training(output)
        workflow(output)
        cv(output)
        operations(output)
        workflow_schemas(output)
        extra_constraints(output)
        bundling_schema(output)
        workflow_extended(output)
        metric_edges(output)
        empty_queries(output)
        final_schemas(output)
        review_refit(output)
        review_cegb(output)
        review_dart(output)
        review_near_zero(output)
        review_schemas(output)
        for family in REVIEW_FAMILIES:
            generated = read_json(output / f'{family}.json')
            jsonschema.Draft202012Validator(read_json(output / f'{family}.schema.json')).validate(generated)
            review_shapes(family, generated, manifest, output)
            compare_exact(generated, read_json(ROOT / f'tests/fixtures/{family}.json'))
        for family in FINAL_FAMILIES:
            generated = read_json(output / f'{family}.json')
            jsonschema.Draft202012Validator(read_json(output / f'{family}.schema.json')).validate(generated)
            final_shapes(family, generated, manifest)
            if family != 'workflow_extended':
                check_generated_executable(family, generated)
            final_compare(family, generated, read_json(ROOT / f'tests/fixtures/{family}.json'))
        check_build_inputs(manifest)
        with tempfile.TemporaryDirectory(prefix='oracle-bundling-', dir=OUT) as bundle_directory:
            bundle_fixtures, generated = bundling(bundle_directory)
            saved = read_json(ROOT / 'tests/fixtures/bundling.json')
            jsonschema.Draft202012Validator(read_json(ROOT / 'tests/fixtures/bundling.schema.json')).validate(generated)
            bundling_shapes(generated, manifest, bundle_fixtures)
            assert bundling_payload(generated) == bundling_payload(saved), 'bundling deterministic payload changed'
            for path in bundle_fixtures.rglob('*.txt'):
                assert path.read_bytes() == (ROOT / 'tests/fixtures' / path.relative_to(bundle_fixtures)).read_bytes(), 'bundling model bytes changed'
        for name, expected in manifest['native_helpers'].items():
            rebuilt = read_json(OUT / f'{name}-record.json')
            assert {k: v for k, v in rebuilt.items() if k != 'executable_sha256'} == {k: v for k, v in expected.items() if k != 'executable_sha256'}, f'{name} build inputs changed'
        for key, filename in [('driver', 'driver-record.json'), ('extended_driver', 'extended-driver-record.json')]:
            rebuilt = read_json(OUT / filename)
            # Mach-O helper bytes can change on relink. Keep each build's hash,
            # but require identical compiler/link inputs and generated evidence.
            expected_inputs = {k: v for k, v in manifest[key].items() if k != 'executable_sha256'}
            assert {k: v for k, v in rebuilt.items() if k != 'executable_sha256'} == expected_inputs, f'{key} build inputs changed'
        for path in sorted(output.rglob("*")):
            if path.is_file():
                saved = ROOT / "tests/fixtures" / path.relative_to(output)
                if path.name in ('metric_edges.json', 'empty_queries.json'):
                    # Only each recorded Mach-O executable hash may differ.
                    final_compare(path.stem, read_json(path), read_json(saved))
                else:
                    assert path.read_bytes() == saved.read_bytes(), f"regeneration differs: {saved}"
    print('Final native families passed: ' + ', '.join(f'{family}={len(names)}' for family, names in FINAL_FAMILIES.items()))
    print('Review native families passed: ' + ', '.join(f'{family}={len(names)}' for family, names in REVIEW_FAMILIES.items()))
    print("Verified loaded image and helper identities; deterministic models, primitives, workflow/CV/operation/extra-constraint evidence and schemas regenerate byte-identically")
    print('Bundling: all 22 native deterministic payloads and model bytes regenerated exactly; runtime resource samples and relink executable hashes checked separately')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", action="store_true", help="verify loaded identity and regenerate in a temporary directory")
    parser.add_argument("--self-test", action="store_true", help="prove deliberate mutations are rejected")
    args = parser.parse_args()
    manifest = validate_saved()
    if args.reference:
        check_reference(manifest)
    if args.self_test:
        self_test(args.reference)


if __name__ == "__main__":
    main()
