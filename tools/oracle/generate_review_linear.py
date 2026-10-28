"""Capture native skipped-tree layout and execute only source-checked safe refits.

A skipped Tree(2, false, false) has no linear vectors. The linear refit path
sets its flag, then indexes those empty vectors. That path is not executed.
NumPy supplies input/output arrays only; all results are actual native outputs.
"""
from pathlib import Path
import time

import numpy as np

from common import COMMIT, ROOT, SOURCE, check_resources, load_reference, save_json, sha

BASE = dict(objective='multiclass', num_class=3, linear_tree=True,
            verbosity=-1, num_threads=1, deterministic=True, force_col_wise=True,
            num_leaves=3, min_data_in_leaf=4, min_data_in_bin=1,
            feature_pre_filter=False, learning_rate=0.1, path_smooth=0.0)


def specifications():
    return [('linear_missing_class', {}), ('linear_all_classes', {}),
            ('constant_missing_class', dict(linear_tree=False)),
            ('linear_constant_features', {}), ('linear_no_features', dict(feature_pre_filter=True)),
            ('linear_binary_single_class', dict(objective='binary', num_class=1)),
            ('linear_cancellation_missing_class', dict(min_data_in_leaf=2)),
            ('linear_cancellation_all_classes', dict(min_data_in_leaf=2))]


NAMES = {name for name, _ in specifications()}


def snapshot(model, rows):
    return dict(num_trees=model.num_trees(), current_iteration=model.current_iteration(),
                raw=np.asarray(model.predict(rows, raw_score=True, num_threads=1)).reshape(len(rows), -1).tolist(),
                transformed=np.asarray(model.predict(rows, num_threads=1)).reshape(len(rows), -1).tolist(),
                leaves=np.asarray(model.predict(rows, pred_leaf=True, num_threads=1)).reshape(len(rows), -1).tolist())


def linear_layout(text):
    return [line == 'is_linear=1' for line in text.splitlines() if line.startswith('is_linear=')]


def generate(fixtures: Path = ROOT / 'tests/fixtures'):
    fixtures = Path(fixtures)
    lgb, record = load_reference()
    assert COMMIT == '5d84f3f222b41409a6983d2ac90513ad9702fa9e'
    assert record['library']['loaded_sha256'] == '9ed1d46d2d60bac2966fd03bd81e7d192025804f6a3e3ef24c3aa483d0ec7c1e'
    model_dir = fixtures / 'models/review_linear'
    model_dir.mkdir(parents=True, exist_ok=True)
    cases = []
    for name, extra in specifications():
        started = time.monotonic()
        params = dict(BASE, **extra)
        rows = [[float(i - 12), float((i * 5) % 11 - 5)] for i in range(24)]
        labels = [int((i * 7) % 13 >= 6) for i in range(24)]
        if name == 'linear_all_classes':
            labels = [(i * 5) % 7 % 3 for i in range(24)]
        elif name == 'linear_cancellation_missing_class':
            labels = [int(i >= 12) for i in range(24)]
        elif name == 'linear_cancellation_all_classes':
            labels = [i % 3 for i in range(24)]
        elif name in ('linear_constant_features', 'linear_no_features'):
            rows = [[1.0, 2.0] for _ in range(24)]
        elif name == 'linear_binary_single_class':
            labels = [0 for _ in range(24)]
        x, y = np.asarray(rows, dtype=np.float64), np.asarray(labels, dtype=np.float32)
        data = lgb.Dataset(x, label=y, params=params)
        fresh = lgb.train(params, data, num_boost_round=3, keep_training_booster=True)
        filename = f'models/review_linear/{name}.txt'
        fresh.save_model(str(fixtures / filename))
        loaded = lgb.Booster(model_file=str(fixtures / filename))
        layout = linear_layout(fresh.model_to_string())
        assert layout == linear_layout(loaded.model_to_string())
        before, reloaded = snapshot(fresh, x), snapshot(loaded, x)
        assert before == reloaded
        structure = [tree['tree_structure'] for tree in fresh.dump_model()['tree_info']]
        refit_labels = [(i * 5) % 7 % 3 for i in range(24)] if name == 'linear_cancellation_all_classes' else labels[::-1]
        refit = dict(status='not_executed_undefined_native_state', rows=rows,
                     labels=refit_labels, decay_rate=0.5,
                     reason='gbdt.cpp:398 creates Tree(2,false,false); io/tree.cpp:54-59 allocates linear vectors only for linear trees; gbdt.cpp:289-300 refits every class; linear_tree_learner.cpp:184 sets the flag only (tree.h:329-331), then line 212 calls unchecked LeafFeatures(0) (tree.h:298) on an empty vector. LeafCoeffs and LeafConst are also unchecked (tree.h:289,292). path_smooth=0 avoids imported leaf_parent access at serial_tree_learner.cpp:268-271.',
                     fresh=None, reloaded=None, model=None)
        if params['linear_tree']:
            # Tree::Tree only allocates linear state when constructed as linear.
            # CalculateLinear::SetIsLinear does not initialize skipped state.
            safe = all(layout)
        else:
            # Serial FitByExistingTree does not use linear state. With smoothing
            # disabled, it does not read imported trees' absent parent vectors.
            safe = True
        if safe:
            results = {}
            for mode, model in (('fresh', fresh), ('reloaded', loaded)):
                result = model.refit(x, np.asarray(refit['labels'], dtype=np.float32),
                                     decay_rate=refit['decay_rate'], dataset_params=params)
                results[mode] = snapshot(result, x)
            assert results['fresh'] == results['reloaded']
            final_file = f'models/review_linear/{name}_refit.txt'
            result.save_model(str(fixtures / final_file))
            saved_refit = lgb.Booster(model_file=str(fixtures / final_file))
            assert snapshot(saved_refit, x) == results['reloaded']
            refit.update(status='executed', reason=None, **results,
                         model=dict(model_file=final_file, model_sha256=sha(fixtures / final_file)))
        cases.append(dict(name=name, params=params, rows=rows, model_file=filename,
                          model_sha256=sha(fixtures / filename),
                          training=dict(rows=rows, labels=labels, num_boost_round=3),
                          native_structure=structure, linear_layout=layout,
                          fresh=before, reloaded=reloaded, refit=refit))
        print(f'{name}: linear_layout={layout}; refit={refit["status"]}')
        check_resources(started)
    assert {case['name'] for case in cases} == NAMES
    sources = ('src/boosting/gbdt.cpp', 'src/boosting/gbdt_model_text.cpp',
               'include/LightGBM/tree.h', 'src/io/tree.cpp',
               'src/treelearner/linear_tree_learner.cpp', 'src/treelearner/linear_tree_learner.h',
               'src/treelearner/serial_tree_learner.cpp', 'src/treelearner/serial_tree_learner.h',
               'src/objective/multiclass_objective.hpp', 'src/objective/binary_objective.hpp',
               'python-package/lightgbm/basic.py',
               'external_libs/eigen/Eigen/src/Core/products/GeneralMatrixVector.h',
               'external_libs/eigen/Eigen/src/Core/arch/NEON/PacketMath.h')
    save_json(fixtures / 'review_linear.json', dict(
        schema_version=1, upstream_commit=COMMIT, source=record['source'],
        library=record['library'], compiler=record['compiler'],
        compile_commands_sha256=record['compile_commands_sha256'],
        producer_sha256=sha(ROOT / 'tools/oracle/generate_review_linear.py'),
        source_sha256={path: sha(SOURCE / path) for path in sources},
        execution='Pinned native CPU serial; NumPy input/output interchange only; no helper executable; skipped-tree linear refit is not executed because source reads empty vectors',
        cases=cases))
    print(f'Generated {len(cases)} actual native review linear cases')


if __name__ == '__main__':
    generate()
