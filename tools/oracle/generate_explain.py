"""Native TreeSHAP and importance from reloaded, unchanged serialized models."""
import json
import time

import numpy as np
from scipy.sparse import csr_matrix

from common import COMMIT, ROOT, SOURCE, check_resources, load_reference, save_json, sha


def generate(fixtures=ROOT / 'tests/fixtures'):
    lgb, _ = load_reference()
    # Regeneration reads models and prediction inputs produced in the same run.
    base = json.loads((fixtures / 'prediction.json').read_text())['cases']
    ranges = json.loads((fixtures / 'prediction_ranges.json').read_text())['cases']
    cases = []
    for original in [*base, *ranges]:
        started = time.monotonic()
        model = lgb.Booster(model_file=str(fixtures / original['model_file']))
        rows = np.asarray([[np.nan if v is None else v for v in row] for row in original['rows']])
        selection = {key: original['params'][key] for key in ('start_iteration', 'num_iteration') if key in original['params']}
        for sparse in (False, True):
            data = csr_matrix(rows) if sparse else rows
            raw = np.asarray(model.predict(data, raw_score=True, num_threads=1, **selection)).reshape(len(rows), -1)
            contributions = model.predict(data, pred_contrib=True, num_threads=1, **selection)
            if isinstance(contributions, list):
                contributions = np.hstack([part.toarray() for part in contributions])
            elif sparse:
                contributions = contributions.toarray()
            contributions = np.asarray(contributions).reshape(len(rows), -1)
            for class_id in range(raw.shape[1]):
                total = contributions[:, class_id * (rows.shape[1] + 1):(class_id + 1) * (rows.shape[1] + 1)].sum(axis=1)
                assert np.all(np.abs(total - raw[:, class_id]) <= 1e-12 + 1e-10 * np.abs(raw[:, class_id])), original['name']
            # Importance has no start_iteration argument. Load the native selected
            # text slice and call the actual C API importance on that model.
            selected = lgb.Booster(model_str=model.model_to_string(**selection)) if selection else model
            split = selected.feature_importance(importance_type='split').tolist()
            gain = selected.feature_importance(importance_type='gain').tolist()
            c = dict(name=original['name'] + ('_csr' if sparse else ''), model_file=original['model_file'],
                     rows=original['rows'], params=original['params'], raw=raw.tolist(),
                     contributions=contributions.tolist(), importance_split=split, importance_gain=gain)
            if sparse:
                c['sparse'] = dict(indptr=data.indptr.tolist(), indices=data.indices.tolist(),
                                  data=[None if np.isnan(v) else float(v) for v in data.data], shape=list(data.shape))
            cases.append(c)
        check_resources(started)
    errors = []
    for original in json.loads((fixtures / 'prediction_linear.json').read_text())['cases']:
        model = lgb.Booster(model_file=str(fixtures / original['model_file']))
        rows = np.asarray([[np.nan if v is None else v for v in row] for row in original['rows']])
        try:
            model.predict(rows, pred_contrib=True, num_threads=1)
        except lgb.basic.LightGBMError as error:
            errors.append(dict(name=original['name'], model_file=original['model_file'], expected_error=str(error)))
        else:
            raise AssertionError('native linear contributions unexpectedly supported')
    model = lgb.Booster(model_file=str(fixtures / base[0]['model_file']))
    loaded_gains = []
    def collect(node):
        if 'split_index' in node:
            loaded_gains.append(node['split_gain'])
            collect(node['left_child'])
            collect(node['right_child'])
    for tree in model.dump_model()['tree_info']:
        collect(tree['tree_structure'])
    save_json(fixtures / 'explain.json', dict(schema_version=1, upstream_commit=COMMIT, cases=cases,
        expected_errors=errors, gain_precision=dict(storage_type='f32', accumulation_type='f64',
            source_file='src/io/tree.cpp', source_sha256=sha(SOURCE / 'src/io/tree.cpp'),
            parser='CommonC::StringToArrayFast<float>', example_model_file=base[0]['model_file'],
            native_loaded_gains=loaded_gains,
            serialized_gain_lines=[line for line in (fixtures / base[0]['model_file']).read_text().splitlines() if line.startswith('split_gain=')])) )
    print(f'Generated {len(cases)} native dense/CSR TreeSHAP cases and {len(errors)} linear errors')


if __name__ == '__main__':
    generate()
