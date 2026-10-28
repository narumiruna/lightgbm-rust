"""Generate finite JSON and unmodified v4 models from the verified reference."""
import numpy as np

from common import COMMIT, ROOT, load_reference, save_json

FIXTURES = ROOT / "tests/fixtures"
BASE = dict(verbosity=-1, device_type="cpu", tree_learner="serial", num_threads=1,
            deterministic=True, force_col_wise=True, seed=0, data_random_seed=0,
            feature_fraction_seed=0, bagging_seed=0, drop_seed=0, extra_seed=0,
            objective_seed=0, num_leaves=3, max_depth=-1, learning_rate=0.25,
            min_data_in_leaf=1, min_data_in_bin=1, min_sum_hessian_in_leaf=0.001,
            max_bin=15, feature_pre_filter=False, boost_from_average=True)


def serial_rows(x):
    return [[None if np.isnan(v) else float(v) for v in row] for row in x]


def generate(fixtures=FIXTURES):
    lgb, record = load_reference()
    (fixtures / "models").mkdir(parents=True, exist_ok=True)
    numeric = np.array([[-3., 0.], [-2., 1.], [-1., 0.], [0., 1.], [1., 0.],
                        [2., 1.], [3., 0.], [4., 1.]], dtype=np.float64)
    probe = np.vstack([numeric, [[np.nan, 0.], [1e-36, 1.], [-1e-36, 0.], [0.5, 1.]]])
    missing = np.array([[-3.], [-2.], [-1.], [0.], [1.], [2.], [np.nan], [np.nan]])
    categorical = np.array([[0.], [1.], [2.], [0.], [1.], [2.], [3.], [3.], [np.nan]])
    specs = [
        ("numeric_l2", numeric, [-2., -1., -1., 0., 1., 2., 2., 3.], probe, "regression", {}, None),
        ("binary", numeric, [0., 0., 0., 0., 1., 1., 1., 1.], probe, "binary", {}, None),
        ("numeric_weighted", numeric, [-2., -1., -1., 0., 1., 2., 2., 3.], probe, "regression", {}, [1., 2., 0.5, 1., 3., 1., 2., 1.]),
        ("missing_nan", missing, [-2., -2., -1., 0., 1., 2., 4., 4.],
         np.vstack([missing, [[1e-36], [-1e-36], [0.5]]]), "regression", {}, None),
        ("missing_zero", missing, [-2., -2., -1., 4., 1., 2., 4., 4.],
         np.vstack([missing, [[1e-36], [-1e-36], [0.5]]]), "regression", {"zero_as_missing": True}, None),
        ("categorical", categorical, [0., 3., 1., 0., 3., 1., 2., 2., -1.],
         np.vstack([categorical, [[-1.], [8.], [1.9], [-0.5]]]), "regression",
         {"categorical_feature": [0], "max_cat_to_onehot": 8, "cat_smooth": 0., "cat_l2": 0.}, None),
        ("multiclass", numeric, [0., 0., 1., 1., 2., 2., 0., 2.], probe, "multiclass", {"num_class": 3}, None),
        ("multiclass_empty_class", numeric, [0., 0., 1., 1., 0., 0., 1., 1.], probe, "multiclass", {"num_class": 3}, None),
        ("single_leaf", np.zeros((4, 1)), [2., 2., 2., 2.], np.array([[0.], [np.nan], [3.]]), "regression", {}, None),
        ("regression_sqrt", numeric, [-2., -1., -1., 0., 1., 2., 2., 3.], probe, "regression", {"reg_sqrt": True}, None),
        ("binary_sigmoid", numeric, [0., 0., 0., 0., 1., 1., 1., 1.], probe, "binary", {"sigmoid": 2.}, None),
        ("multiclassova", numeric, [0., 0., 1., 1., 2., 2., 0., 2.], probe, "multiclassova", {"num_class": 3}, None),
        ("poisson", numeric, [1., 1., 2., 2., 3., 4., 4., 5.], probe, "poisson", {}, None),
        ("gamma", numeric, [1., 1., 2., 2., 3., 4., 4., 5.], probe, "gamma", {}, None),
        ("tweedie", numeric, [0., 1., 2., 2., 3., 4., 4., 5.], probe, "tweedie", {"tweedie_variance_power": 1.5}, None),
        ("cross_entropy", numeric, [0., 0.1, 0.2, 0.3, 0.7, 0.8, 0.9, 1.], probe, "cross_entropy", {}, None),
        ("cross_entropy_lambda", numeric, [0., 0.1, 0.2, 0.3, 0.7, 0.8, 0.9, 1.], probe, "cross_entropy_lambda", {}, None),
        ("rf_average", numeric, [-2., -1., -1., 0., 1., 2., 2., 3.], probe, "regression",
         {"boosting": "rf", "bagging_fraction": 0.8, "bagging_freq": 1, "feature_fraction": 0.8}, None),
        ("missing_none", missing, [-2., -2., -1., 0., 1., 2., 4., 4.],
         np.vstack([missing, [[1e-36], [-1e-36], [0.5]]]), "regression", {"use_missing": False}, None),
    ]
    cases = []
    for name, x, labels, rows, objective, extra, weights in specs:
        params = dict(BASE, objective=objective, **extra)
        # Dataset receives f64 features but f32 labels/weights, as in the upstream default build.
        kwargs = {}
        if "categorical_feature" in extra:
            kwargs["categorical_feature"] = extra["categorical_feature"]
        ds = lgb.Dataset(x, label=np.asarray(labels, dtype=np.float32),
                         weight=None if weights is None else np.asarray(weights, dtype=np.float32),
                         params=params, **kwargs)
        booster = lgb.train(params, ds, num_boost_round=3)
        model_file = f"models/{name}.txt"
        booster.save_model(fixtures / model_file)
        # Re-load serialized model: outputs describe exactly the bytes saved to fixtures.
        model = lgb.Booster(model_file=str(fixtures / model_file))
        # Probe both sides of each serialized numerical threshold, not an approximate decimal.
        boundary_rows = []
        def boundaries(node):
            if "split_index" not in node:
                return
            if node["decision_type"] == "<=":
                threshold = float(node["threshold"])
                if np.isfinite(threshold):
                    for v in (np.nextafter(threshold, -np.inf), threshold, np.nextafter(threshold, np.inf)):
                        row = np.zeros(x.shape[1], dtype=np.float64)
                        row[node["split_feature"]] = v
                        boundary_rows.append(row)
            boundaries(node["left_child"])
            boundaries(node["right_child"])
        for tree in model.dump_model()["tree_info"]:
            boundaries(tree["tree_structure"])
        if "categorical_feature" not in extra:
            zero = float(np.float32(1e-35))
            for v in (np.nextafter(-zero, -np.inf), -zero, np.nextafter(-zero, np.inf),
                      np.nextafter(zero, -np.inf), zero, np.nextafter(zero, np.inf)):
                row = np.zeros(x.shape[1], dtype=np.float64)
                row[0] = v
                boundary_rows.append(row)
        if boundary_rows:
            rows = np.vstack([rows, boundary_rows])
        def pred(**kw):
            return np.asarray(model.predict(rows, num_threads=1, **kw)).reshape(len(rows), -1).tolist()
        training = dict(rows=serial_rows(x), labels=np.asarray(labels, dtype=np.float32).tolist(),
                        num_boost_round=3, expected_structure=[t["tree_structure"] for t in model.dump_model()["tree_info"]])
        if weights is not None:
            training["weights"] = np.asarray(weights, dtype=np.float32).tolist()
        cases.append(dict(name=name, model_file=model_file, rows=serial_rows(rows), raw=pred(raw_score=True),
                          transformed=pred(), leaves=np.asarray(pred(pred_leaf=True), dtype=np.int64).tolist(),
                          params=params, training=training))
    main_cases = [c for c in cases if c["name"] != "multiclass_empty_class"]
    ranges = [c for c in cases if c["name"] == "multiclass_empty_class"]
    for original in main_cases:
        if original["name"] not in ("multiclass", "rf_average"):
            continue
        model = lgb.Booster(model_file=str(fixtures / original["model_file"]))
        rows = np.array([[np.nan if v is None else v for v in row] for row in original["rows"]])
        for count in (1, 2):
            def prediction(**kwargs):
                return np.asarray(model.predict(rows, start_iteration=1, num_iteration=count,
                                                num_threads=1, **kwargs)).reshape(len(rows), -1).tolist()
            ranges.append(dict(name=f"{original['name']}_start1_count{count}", model_file=original["model_file"],
                               rows=original["rows"], raw=prediction(raw_score=True), transformed=prediction(),
                               leaves=np.asarray(prediction(pred_leaf=True), dtype=np.int64).tolist(),
                               params=dict(original["params"], start_iteration=1, num_iteration=count)))
    save_json(fixtures / "prediction.json", dict(schema_version=1, upstream_commit=COMMIT, cases=main_cases))
    save_json(fixtures / "prediction_ranges.json", dict(schema_version=1, upstream_commit=COMMIT, cases=ranges))
    print(f"Generated {len(main_cases)} prediction/training and {len(ranges)} range/empty-class cases; loaded {record['library']}")


if __name__ == "__main__":
    generate()
