"""Native regression evidence for independently reviewed training edge cases."""
import numpy as np
from common import COMMIT, ROOT, load_reference, save_json
from generate_predictions import BASE, serial_rows

def generate(fixtures=ROOT / "tests/fixtures"):
    lgb, _ = load_reference()
    specs = [
        ("missing_only", [-1., 1., np.nan, np.nan], [0., 0., 1., 1.], {}),
        ("missing_direction_tie", [1., np.nan, 0., 2., 1., -1.], [1., -2., 0., 1., 2., -2.],
         dict(max_bin=7, lambda_l2=0.5, max_delta_step=1., learning_rate=0.3)),
    ]
    (fixtures / "models").mkdir(parents=True, exist_ok=True)
    cases = []
    for name, values, labels, extra in specs:
        x = np.asarray(values, dtype=np.float64).reshape(-1, 1)
        params = dict(BASE, objective="regression", **extra)
        data = lgb.Dataset(x, label=np.asarray(labels, dtype=np.float32), params=params)
        model = lgb.train(params, data, num_boost_round=1)
        filename = f"models/{name}.txt"
        model.save_model(fixtures / filename)
        model = lgb.Booster(model_file=str(fixtures / filename))
        rows = np.vstack([x, [[-1e301], [1e301], [0.]]])
        def pred(**kwargs):
            return np.asarray(model.predict(rows, num_threads=1, **kwargs)).reshape(len(rows), -1).tolist()
        cases.append(dict(name=name, model_file=filename, rows=serial_rows(rows),
                          raw=pred(raw_score=True), transformed=pred(),
                          leaves=np.asarray(pred(pred_leaf=True), dtype=np.int64).tolist(), params=params,
                          training=dict(rows=serial_rows(x), labels=np.asarray(labels,dtype=np.float32).tolist(),
                                        num_boost_round=1, expected_structure=[t["tree_structure"] for t in model.dump_model()["tree_info"]])))
    save_json(fixtures / "prediction_regressions.json", dict(schema_version=1, upstream_commit=COMMIT, cases=cases))
    print("Generated actual native missing-only and missing-direction tie regressions")

if __name__ == "__main__":
    generate()
