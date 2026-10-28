"""Retain and run the original native BinMapper feature-info helper."""
import math
import shlex
import subprocess

from common import COMMIT, ROOT, save_json
from native import compile_helper


def specifications():
    return [dict(name=name, rows=rows, categorical=categorical) for name, rows, categorical in [
        ('integers', [-3., 0., 1., 5.], False),
        ('decimal', [-0.1, 0.7], False),
        ('scientific_small', [1e-5, 2e-5], False),
        ('fixed_small', [1e-4, 2e-4], False),
        ('fixed_large', [1e16, 2e16], False),
        ('scientific_large', [1e17, 2e17], False),
        ('all_nan', [None, None], False),
        ('zeros', [-0., 0., 0.], False),
        ('adjacent', [1., math.nextafter(1., math.inf), math.nextafter(math.nextafter(1., math.inf), math.inf)], False),
        ('negative_and_implicit_zero', [-3., -1., 0.], False),
        ('infinite_extrema', ['-Infinity', 'Infinity', None, 0.], False),
        ('categories', [0., 1., 1., 2., -1., None], True),
        ('categories_all_missing', [None, None], True),
        ('categories_all_negative', [-1., -2.], True),
    ]]


def generate(fixtures=ROOT / 'tests/fixtures'):
    executable, _ = compile_helper('data_feature_info')
    lines = subprocess.check_output([str(executable)], text=True, timeout=300).splitlines()
    cases = specifications()
    names = {c['name'] for c in cases}
    outputs = dict(shlex.split(line) for line in lines if line.split(' ', 1)[0] in names)
    assert set(outputs) == {c['name'] for c in cases}
    for c in cases:
        c.update(expected=outputs[c['name']], params=dict(max_bin=63, min_data_in_bin=1,
                 min_split_data=1, pre_filter=False, use_missing=True, zero_as_missing=False,
                 forced_upper_bounds=[]))
    save_json(fixtures / 'data_feature_info.json', dict(schema_version=1, upstream_commit=COMMIT, cases=cases))
    print(f'Generated {len(cases)} native feature-info strings')


if __name__ == '__main__':
    generate()
