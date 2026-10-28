"""Capture original cv(), source-selected folds, and native per-fold histories."""
import time

import numpy as np

from common import COMMIT, ROOT, check_resources, load_reference, save_json
from generate_predictions import BASE
from workflow_common import integer_data, metrics, provenance, save_model

NAMES = {'explicit', 'standard_seeded', 'standard_unshuffled', 'stratified_seeded', 'group_safe', 'early_truncation'}


def generate(fixtures=ROOT / 'tests/fixtures'):
    lgb, record = load_reference()
    (fixtures / 'models').mkdir(parents=True, exist_ok=True)
    x, original = integer_data()
    cases = []
    for name in sorted(NAMES):
        started = time.monotonic()
        seed, nfold, rounds = 19, 3, 6
        labels = [int(y > 0) for y in original] if name == 'stratified_seeded' else original
        params = dict(BASE, objective='binary' if name == 'stratified_seeded' else 'regression', metric=['binary_logloss', 'auc'] if name == 'stratified_seeded' else ['l2', 'l1'])
        groups = None
        if name == 'group_safe':
            labels = [3, 0, 2, 1] * 6
            groups = [4] * 6
            params.update(objective='lambdarank', metric=['ndcg', 'map'], eval_at=[1, 3])
        data = lgb.Dataset(x, label=np.asarray(labels, dtype=np.float32), group=groups, params=params, free_raw_data=False)
        source_folds, actual_folds, steps = [], [], []
        folds = None
        stratified = name == 'stratified_seeded'
        shuffle = name != 'standard_unshuffled'
        if name == 'explicit':
            for fold in range(nfold):
                valid = [i for i in range(len(x)) if i % nfold == fold]
                train = [i for i in range(len(x)) if i not in valid]
                source_folds.append(dict(train_indices=train, valid_indices=valid))
            folds = [(np.asarray(f['train_indices']), np.asarray(f['valid_indices'])) for f in source_folds]
        generator = ('explicit arrays' if folds is not None else 'sklearn.model_selection.GroupKFold(n_splits=3, shuffle=False)' if groups else
                     'sklearn.model_selection.StratifiedKFold(n_splits=3, shuffle=True, random_state=19)' if stratified else
                     'numpy.random.RandomState(19).permutation + numpy.array_split' if shuffle else 'numpy.arange + numpy.array_split')
        # Observe splitter outputs; delegate every split to the original source-selected generator.
        engine = lgb.engine
        splitter_key = '_LGBMGroupKFold' if groups else '_LGBMStratifiedKFold'
        original_splitter = getattr(engine, splitter_key)
        class ObserveSplitter:
            def __init__(self, *args, **kwargs):
                self.delegate = original_splitter(*args, **kwargs)
            def split(self, *args, **kwargs):
                for train, valid in self.delegate.split(*args, **kwargs):
                    source_folds.append(dict(train_indices=train.tolist(), valid_indices=valid.tolist()))
                    yield train, valid
        def preprocess(train, valid, fold_params):
            actual_folds.append(dict(train_indices=list(map(int, train.used_indices)), valid_indices=list(map(int, valid.used_indices))))
            return train, valid, fold_params
        def capture(env):
            steps.append(dict(iteration=env.iteration, aggregate=metrics(env.evaluation_result_list),
                              folds=[dict(evaluation=metrics(b.eval_valid()), current_iteration=b.current_iteration()) for b in env.model.boosters]))
        capture.order = 5
        capture.before_iteration = False
        early = lgb.early_stopping(2, verbose=False, min_delta=1e6) if name == 'early_truncation' else None
        callbacks = [capture] + ([] if early is None else [early])
        if stratified or groups:
            setattr(engine, splitter_key, ObserveSplitter)
        try:
            result = lgb.cv(params, data, num_boost_round=rounds, folds=folds, nfold=nfold, stratified=stratified,
                            shuffle=shuffle, seed=seed, fpreproc=preprocess, eval_train_metric=True,
                            return_cvbooster=True, callbacks=callbacks)
        finally:
            setattr(engine, splitter_key, original_splitter)
        permutation = None
        if not stratified and not groups and folds is None:
            permutation = np.random.RandomState(seed).permutation(len(x)).tolist() if shuffle else list(range(len(x)))
            parts = np.array_split(np.asarray(permutation), nfold)
            for fold in range(nfold):
                source_folds.append(dict(train_indices=np.concatenate([p for i, p in enumerate(parts) if i != fold]).tolist(), valid_indices=parts[fold].tolist()))
        assert actual_folds == [dict(train_indices=sorted(f['train_indices']), valid_indices=sorted(f['valid_indices'])) for f in source_folds]
        cvbooster = result.pop('cvbooster')
        cv_path = f'models/cv_{name}_booster.json'
        cvbooster.save_model(fixtures / cv_path)
        models = []
        for fold, model in enumerate(cvbooster.boosters):
            models.append(dict(best_iteration=model.best_iteration,
                               final_model=save_model(model, fixtures, f'cv_{name}_fold{fold}', x),
                               full_model=save_model(model, fixtures, f'cv_{name}_fold{fold}_full', x, num_iteration=-1)))
        cases.append(dict(name=name, rows=x.tolist(), labels=labels, groups=groups, params=params,
                          num_boost_round=rounds, seed=seed, nfold=nfold, stratified=stratified, shuffle=shuffle,
                          fold_generator=generator, permutation=permutation, source_folds=source_folds, folds=actual_folds,
                          aggregate_std='population; original engine numpy.std ddof=0', steps=steps, history=result,
                          stopping=None if early is None else dict(stopping_rounds=2, min_delta=1e6),
                          best_iteration=cvbooster.best_iteration, cvbooster_file=cv_path, models=models))
        check_resources(started)
    save_json(fixtures / 'cv.json', dict(schema_version=1, upstream_commit=COMMIT, provenance=provenance(record), cases=cases))
    print(f'Generated {len(cases)} actual cv cases with observed fold arrays and native fold histories')


if __name__ == '__main__':
    generate()
