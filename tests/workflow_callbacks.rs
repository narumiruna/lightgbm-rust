use lightgbm_rust::workflow::{
    CallbackAction, CallbackEnv, EarlyStopping, FitOptions, LearningRateSchedule, TrainingCallback,
};
use lightgbm_rust::{Config, Dataset, Error, Result, Trainer};
use serde_json::Value;
use std::{cell::RefCell, rc::Rc};

struct Observe {
    name: &'static str,
    order: i32,
    before: bool,
    events: Rc<RefCell<Vec<Value>>>,
}
impl TrainingCallback for Observe {
    fn order(&self) -> i32 {
        self.order
    }
    fn before_iteration(&self) -> bool {
        self.before
    }
    fn call(&mut self, env: &CallbackEnv<'_>) -> Result<CallbackAction> {
        self.events.borrow_mut().push(serde_json::json!({"callback":self.name,"iteration":env.iteration,"begin_iteration":env.begin_iteration,"end_iteration":env.end_iteration,"model_current_iteration":env.model_current_iteration,"model_best_iteration":env.best_iteration,"learning_rate":env.config.learning_rate,"history_length":env.evaluation_history.len(),"evaluation":env.evaluation.map(|e|e.iter().map(|v|serde_json::json!({"dataset_name":v.dataset,"metric_name":v.metric,"metric_value":v.value,"maximize":v.higher_is_better,"metric_std_dev":null})).collect::<Vec<_>>())}));
        Ok(CallbackAction::Continue)
    }
}
#[test]
fn ordered_read_only_state_matches_source_callback_events() {
    let fixture: Value = serde_json::from_str(include_str!("fixtures/workflow.json")).unwrap();
    for name in ["ordering_schedule", "early_multival"] {
        let c = fixture["cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|c| c["name"] == name)
            .unwrap();
        let mut config =
            Config::from_pairs(c["params"].as_object().unwrap().iter().map(|(k, v)| {
                (
                    k.clone(),
                    v.as_array()
                        .map(|a| {
                            a.iter()
                                .map(|x| x.as_str().unwrap())
                                .collect::<Vec<_>>()
                                .join(",")
                        })
                        .unwrap_or_else(|| {
                            v.as_str()
                                .map(str::to_owned)
                                .unwrap_or_else(|| v.to_string())
                        }),
                )
            }))
            .unwrap();
        config.num_iterations = 7;
        let rows: Vec<Vec<f64>> = serde_json::from_value(c["rows"].clone()).unwrap();
        let flat: Vec<_> = rows.iter().flatten().copied().collect();
        let data = Dataset::from_dense(
            &flat,
            24,
            2,
            serde_json::from_value(c["labels"].clone()).unwrap(),
            &config,
        )
        .unwrap();
        let mut validation = Vec::new();
        for v in c["valid_sets"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|v| !v["is_training"].as_bool().unwrap())
        {
            validation.push((
                v["name"].as_str().unwrap(),
                Dataset::validation_from_dense(
                    &data,
                    &flat,
                    24,
                    2,
                    serde_json::from_value(v["labels"].clone()).unwrap(),
                )
                .unwrap(),
            ));
        }
        let events = Rc::new(RefCell::new(Vec::new()));
        let mut options = FitOptions {
            evaluate_training: true,
            training_name: "train_alias".into(),
            ..FitOptions::default()
        };
        // Deliberately insert callbacks in reverse order.
        for (name, order, before) in [
            ("after_stopping", 40, false),
            ("after_history", 25, false),
            ("after_first", -10, false),
            ("before_last", 15, true),
            ("before_first", -20, true),
        ] {
            options.callbacks.push(Box::new(Observe {
                name,
                order,
                before,
                events: events.clone(),
            }));
        }
        if name == "ordering_schedule" {
            options
                .callbacks
                .push(Box::new(LearningRateSchedule::values(
                    serde_json::from_value(c["learning_rate_schedule"].clone()).unwrap(),
                )));
        } else {
            options.callbacks.push(Box::new(EarlyStopping::new(2)));
        }
        Trainer::new(config)
            .unwrap()
            .fit_with_options(
                &data,
                &validation.iter().map(|(n, d)| (*n, d)).collect::<Vec<_>>(),
                options,
            )
            .unwrap();
        let expected = c["events"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|e| {
                e["phase"] == "enter"
                    && [
                        "before_first",
                        "before_last",
                        "after_first",
                        "after_history",
                        "after_stopping",
                    ]
                    .contains(&e["callback"].as_str().unwrap())
            })
            .collect::<Vec<_>>();
        let actual = events.borrow();
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            for key in [
                "callback",
                "iteration",
                "begin_iteration",
                "end_iteration",
                "model_current_iteration",
                "model_best_iteration",
                "learning_rate",
                "evaluation",
            ] {
                assert_eq!(actual[key], expected[key], "{name}: {key}");
            }
            let history_len = expected["history_lengths"]
                .as_object()
                .unwrap()
                .values()
                .flat_map(|v| v.as_object().unwrap().values())
                .map(|v| v.as_u64().unwrap())
                .max()
                .unwrap_or(0);
            assert_eq!(
                actual["history_length"].as_u64().unwrap(),
                history_len,
                "{name}"
            );
        }
    }
}
struct Fail;
impl TrainingCallback for Fail {
    fn call(&mut self, _: &CallbackEnv<'_>) -> Result<CallbackAction> {
        Err(Error::InvalidInput("callback failure".into()))
    }
}
#[test]
fn callback_failure_returns_no_partial_model() {
    let config = Config {
        min_data_in_bin: 1,
        min_data_in_leaf: 1,
        ..Config::default()
    };
    let data = Dataset::from_dense(&[0., 1., 2., 3.], 4, 1, vec![0., 1., 2., 3.], &config).unwrap();
    assert!(
        Trainer::new(config.clone())
            .unwrap()
            .fit_with_options(
                &data,
                &[],
                FitOptions {
                    callbacks: vec![Box::new(Fail)],
                    ..FitOptions::default()
                }
            )
            .is_err()
    );
    assert!(Trainer::new(config).unwrap().fit(&data).is_ok());
}
