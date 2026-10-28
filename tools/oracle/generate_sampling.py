"""Run actual native row/query/feature/GOSS/DART and quantization internals."""
import copy
import json
import subprocess

from common import COMMIT, OUT, ROOT, save_json
from native import compile_helper


def specifications(fixtures=ROOT / 'tests/fixtures'):
    trained = json.loads((fixtures / 'prediction_boosting.json').read_text())['cases']
    cases = []
    for c in trained:
        name = c['name']
        if name.startswith('rf_'):
            continue
        kind = 'dart' if name.startswith('dart_') else 'features' if name.startswith('feature_') else 'rows'
        train = c['training']
        case = dict(name=name, kind=kind, params=c['params'], rows=train['rows'], labels=train['labels'],
                    weights=train['weights'], num_steps=9)
        for key in ('groups', 'positions'):
            if key in train:
                case[key] = train[key]
        if kind == 'rows':
            n = len(train['labels'])
            case['scores'] = [float(i) / 8 - 2 + k * 0.3 for k in range(c['params'].get('num_class', 1)) for i in range(n)]
        cases.append(case)
    filtered = copy.deepcopy(next(c for c in cases if c['name'] == 'feature_tree_node'))
    filtered['name'] = 'feature_filtered'
    for row in filtered['rows']:
        row[1] = 0.
    cases.append(filtered)
    template = trained[0]['training']
    for stochastic in (False, True):
        for constant in (False, True):
            params = dict(trained[0]['params'])
            params.update(bagging_fraction=1., bagging_freq=0, num_grad_quant_bins=8, stochastic_rounding=stochastic, seed=42)
            cases.append(dict(name='quantization_' + ('stochastic' if stochastic else 'deterministic') + ('_constant' if constant else ''),
                kind='quantization', params=params, rows=template['rows'], labels=template['labels'],
                num_steps=4, constant_hessian=constant,
                gradients=[(i - 15.25) / 7 for i in range(32)], hessians=[1. if constant else (i % 7 + 1) / 8 for i in range(32)]))
    return dict(schema_version=1, upstream_commit=COMMIT, cases=cases)


def generate(fixtures=ROOT / 'tests/fixtures'):
    executable, _ = compile_helper('sampling_driver')
    request = OUT / 'sampling-request.json'
    save_json(request, specifications(fixtures))
    subprocess.run([str(executable), str(request), str(fixtures)], check=True, timeout=300)
    path = fixtures / 'sampling.json'
    output = json.loads(path.read_text())
    save_json(path, output)
    print(f"Generated {len(output['cases'])} native internal sampling/quantization cases")


if __name__ == '__main__':
    generate()
