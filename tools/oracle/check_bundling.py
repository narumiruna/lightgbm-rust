"""Validate native bundling evidence without changing the separately owned producer."""
import copy
import json
from pathlib import Path

from common import COMMIT, ROOT, save_json, sha

NAMES = {name + '_' + enabled for name in ('exclusive', 'sparse', 'conflict', 'missing', 'zero_missing',
         'frequent_nonzero', 'filtered', 'categorical', 'categorical_nan', 'reordered', 'extra_trees')
         for enabled in ('true', 'false')}


def schema():
    def obj(properties):
        return dict(type='object', properties=properties, required=list(properties), additionalProperties=False)
    def array(items):
        return dict(type='array', minItems=1, items=items)
    integer = dict(type='integer', minimum=0)
    number = dict(type='number')
    digest = dict(type='string', pattern='^[a-f0-9]{64}$')
    optional = lambda value: dict(anyOf=[value, dict(type='null')])
    helper = json.loads((ROOT / 'docs/reference-manifest.schema.json').read_text())['properties']['native_helpers']['additionalProperties']
    case = obj(dict(name=dict(enum=sorted(NAMES)), rows=array(array(optional(number))),
                    parameters=dict(type='string', minLength=1), params=dict(type='object', minProperties=1),
                    bins=array(array(optional(integer))), inner_features=array(dict(type='integer', minimum=-1)),
                    maps=array(optional(obj(dict(default_bin=integer, most_freq_bin=integer, missing_type=dict(enum=[0, 1, 2]),
                                                 num_bins=dict(type='integer', minimum=1), group=integer)))),
                    groups=array(obj(dict(features=array(integer), multi_value=dict(type='boolean'), decoded_conflicts=integer))),
                    labels=array(number), model=dict(type='string', pattern=r'^bundling_models/[a-z_]+\.txt$'),
                    model_sha256=digest, raw_predictions=array(number)))
    library = obj(dict(path=dict(type='string'), sha256=digest, loaded_path=dict(type='string'), loaded_sha256=digest,
                       verification=dict(const='dladdr(LGBM_GetLastError) + SHA-256')))
    return dict({'$schema': 'https://json-schema.org/draft/2020-12/schema'}, **obj(dict(
        schema_version=dict(const=1), upstream_commit=dict(const=COMMIT), loaded_library=library,
        native_helper=helper, resources=obj(dict(peak_rss_bytes=dict(type='integer', minimum=1, exclusiveMaximum=512 * 1024 * 1024),
                                                elapsed_seconds=dict(type='number', minimum=0, exclusiveMaximum=300))),
        cases=dict(type='array', minItems=22, maxItems=22, items=case))))


def stable_payload(data):
    result = copy.deepcopy(data)
    del result['resources']
    del result['native_helper']['executable_sha256']
    return result


def validate(data, manifest=None, fixtures=ROOT / 'tests/fixtures'):
    helper = data['native_helper']
    assert helper['source_sha256'] == sha(ROOT / 'tools/oracle/bundling_driver.cpp')
    if manifest:
        assert data['loaded_library'] == {k: v for k, v in manifest['loaded_library'].items() if k != 'linked_libraries'}
        assert {k: v for k, v in helper.items() if k != 'executable_sha256'} == {
            k: v for k, v in manifest['native_helpers']['bundling_driver'].items() if k != 'executable_sha256'}
    assert helper['loaded_library_sha256'] == data['loaded_library']['loaded_sha256']
    cases = {c['name']: c for c in data['cases']}
    assert len(cases) == len(data['cases']) and set(cases) == NAMES
    for c in data['cases']:
        n, features = len(c['rows']), len(c['rows'][0])
        assert len(c['labels']) == len(c['raw_predictions']) == len(c['bins']) == n
        assert len(c['inner_features']) == len(c['maps']) == features
        assert all(len(row) == features for row in c['rows'] + c['bins'])
        eligible = [f for f, inner in enumerate(c['inner_features']) if inner >= 0]
        assert sorted(c['inner_features'][f] for f in eligible) == list(range(len(eligible)))
        assert sorted(f for group in c['groups'] for f in group['features']) == eligible
        assert c['params']['enable_bundle'] == c['name'].endswith('_true')
        if not c['params']['enable_bundle']:
            assert all(len(group['features']) == 1 for group in c['groups'])
        for f, mapping in enumerate(c['maps']):
            if f not in eligible:
                assert mapping is None and all(row[f] is None for row in c['bins'])
                continue
            assert mapping['default_bin'] < mapping['num_bins'] and mapping['most_freq_bin'] < mapping['num_bins']
            assert 0 <= mapping['group'] < len(c['groups'])
            assert f in c['groups'][mapping['group']]['features']
            assert all(type(row[f]) is int and 0 <= row[f] < mapping['num_bins'] for row in c['bins'])
        for group in c['groups']:
            conflicts = sum(max(0, sum(row[f] != c['maps'][f]['most_freq_bin'] for f in group['features']) - 1) for row in c['bins'])
            assert conflicts == group['decoded_conflicts']
        path = fixtures / c['model']
        assert path.resolve().is_relative_to(fixtures / 'bundling_models') and sha(path) == c['model_sha256']
        assert sum(line.startswith('Tree=') for line in path.read_text().splitlines()) == 2


def regenerate(directory):
    # The active producer has no output-directory argument. Redirect only its
    # output ROOT; native compilation/loading still use common.ROOT unchanged.
    import generate_bundling as producer
    root = Path(directory)
    fixtures = root / 'tests/fixtures'
    fixtures.mkdir(parents=True)
    original = producer.ROOT
    try:
        producer.ROOT = root
        data = producer.generate()
    finally:
        producer.ROOT = original
    return fixtures, data


def generate_schema(fixtures=ROOT / 'tests/fixtures'):
    save_json(fixtures / 'bundling.schema.json', schema())


if __name__ == '__main__':
    generate_schema()
