//! Adapter tests use native fixture values, without Python or native libraries.
use lightgbm_rust::io::{InputFormat, InputShape, load_dataset, load_refit_dataset, predict};
use lightgbm_rust::{Config, Model};
use serde_json::Value;
use std::io::{self, Read};
use std::path::Path;

fn config() -> Config {
    Config::from_pairs([
        ("min_data_in_leaf", "1"),
        ("min_data_in_bin", "1"),
        ("feature_pre_filter", "false"),
    ])
    .unwrap()
}
fn shape(rows: usize, features: usize) -> InputShape {
    InputShape::new(rows, features).unwrap()
}

#[test]
fn fixture_training_rows_preserve_labels_and_feature_indexes_in_all_formats() {
    let suite: Value = serde_json::from_str(include_str!("fixtures/prediction.json")).unwrap();
    for case in
        suite["cases"].as_array().unwrap().iter().filter(|c| {
            ["numeric_l2", "binary", "missing_nan"].contains(&c["name"].as_str().unwrap())
        })
    {
        let rows = case["training"]["rows"].as_array().unwrap();
        let labels = case["training"]["labels"].as_array().unwrap();
        let columns = rows[0].as_array().unwrap().len();
        let mut datasets = Vec::new();
        for format in [InputFormat::Csv, InputFormat::Tsv, InputFormat::LibSvm] {
            let text = rows
                .iter()
                .zip(labels)
                .map(|(row, label)| {
                    let row = row.as_array().unwrap();
                    match format {
                        InputFormat::LibSvm => format!(
                            "{} {}",
                            label,
                            row.iter()
                                .enumerate()
                                .rev()
                                .map(|(i, v)| format!(
                                    "{}:{}",
                                    i + 1,
                                    v.as_f64()
                                        .map(|v| v.to_string())
                                        .unwrap_or_else(|| "NaN".into())
                                ))
                                .collect::<Vec<_>>()
                                .join(" ")
                        ),
                        _ => {
                            let delimiter = if format == InputFormat::Csv {
                                ","
                            } else {
                                "\t"
                            };
                            format!(
                                "{}{delimiter}{}",
                                label,
                                row.iter()
                                    .map(|v| v.as_f64().map(|v| v.to_string()).unwrap_or_default())
                                    .collect::<Vec<_>>()
                                    .join(delimiter)
                            )
                        }
                    }
                })
                .collect::<Vec<_>>()
                .join("\n");
            let dataset = load_dataset(
                text.as_bytes(),
                format,
                shape(rows.len(), columns),
                &config(),
            )
            .unwrap();
            assert_eq!(
                dataset.labels(),
                labels
                    .iter()
                    .map(|v| v.as_f64().unwrap() as f32)
                    .collect::<Vec<_>>()
            );
            assert_eq!(dataset.ncols(), columns);
            datasets.push(dataset);
        }
        for dataset in &datasets[1..] {
            for r in 0..rows.len() {
                for c in 0..columns {
                    assert_eq!(
                        dataset.get_bin(r, c).unwrap(),
                        datasets[0].get_bin(r, c).unwrap()
                    );
                }
            }
        }
    }
}

#[test]
fn quoted_numeric_csv_missing_values_and_sparse_implicit_zero() {
    let dense = load_dataset(
        "0,\"-3\",0\r\n1,,1\r\n".as_bytes(),
        InputFormat::Csv,
        shape(2, 2),
        &config(),
    )
    .unwrap();
    let sparse = load_dataset(
        "0 1:-3\n1 2:1 1:NaN\n".as_bytes(),
        InputFormat::LibSvm,
        shape(2, 2),
        &config(),
    )
    .unwrap();
    for r in 0..2 {
        for c in 0..2 {
            assert_eq!(dense.get_bin(r, c).unwrap(), sparse.get_bin(r, c).unwrap());
        }
    }
}

#[test]
fn invalid_shapes_rows_fields_labels_and_sparse_entries_fail() {
    for (r, c) in [(0, 1), (1, 0), (usize::MAX, 1), (1, usize::MAX)] {
        assert!(InputShape::new(r, c).is_err());
    }
    assert!("parquet".parse::<InputFormat>().is_err());
    for text in [
        "",
        "0,1\n",
        "0,1,2,3\n",
        "0,1,2\n0,3,4\n",
        "label,a,b\n",
        "NaN,1,2\n",
        "inf,1,2\n",
        "1e40,1,2\n",
        "0,inf,2\n",
        "0,1e999,2\n",
        "0,word,2\n",
    ] {
        assert!(
            load_dataset(text.as_bytes(), InputFormat::Csv, shape(1, 2), &config()).is_err(),
            "accepted {text:?}"
        );
    }
    for text in [
        "",
        "\n",
        "0 0:1\n",
        "0 3:1\n",
        "0 1:2 1:3\n",
        "0 2:2 1:3 2:4\n",
        "0 -1:2\n",
        "0 999999999999999999999999999:1\n",
        "0 1:inf\n",
        "NaN 1:2\n",
        "1e40 1:2\n",
        "0 1\n",
        "0 1:2:3\n",
        "0 qid:1\n",
        "0 1:2 #comment\n",
        "0 1:2\n\n",
    ] {
        assert!(
            load_dataset(text.as_bytes(), InputFormat::LibSvm, shape(1, 2), &config()).is_err(),
            "accepted {text:?}"
        );
    }
    let binary = Config::from_pairs([("objective", "binary")]).unwrap();
    for label in ["-1", "0.5", "2"] {
        assert!(
            load_dataset(
                format!("{label},1,2\n").as_bytes(),
                InputFormat::Csv,
                shape(1, 2),
                &binary
            )
            .is_err()
        );
    }
}

#[test]
fn prediction_validates_shape_and_libsvm_label_without_emitting_labels() {
    let model = Model::load(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/models/numeric_l2.txt"),
    )
    .unwrap();
    let mut dense = Vec::new();
    predict(
        "-3,0\n".as_bytes(),
        InputFormat::Csv,
        shape(1, 2),
        &model,
        &mut dense,
    )
    .unwrap();
    let mut sparse = Vec::new();
    predict(
        "123 1:-3\n".as_bytes(),
        InputFormat::LibSvm,
        shape(1, 2),
        &model,
        &mut sparse,
    )
    .unwrap();
    assert_eq!(dense, sparse);
    assert!(!String::from_utf8(dense).unwrap().contains("123"));
    for (text, format, rows, cols) in [
        ("-3,0\n", InputFormat::Csv, 2, 2),
        ("-3,0\n", InputFormat::Csv, 1, 3),
        ("NaN 1:-3\n", InputFormat::LibSvm, 1, 2),
        ("1 1:-3 1:0\n", InputFormat::LibSvm, 1, 2),
    ] {
        assert!(
            predict(
                text.as_bytes(),
                format,
                shape(rows, cols),
                &model,
                Vec::new()
            )
            .is_err()
        );
    }
}

struct TooMuch;
impl Read for TooMuch {
    fn read(&mut self, output: &mut [u8]) -> io::Result<usize> {
        output.fill(b'0');
        Ok(output.len())
    }
}
#[test]
fn input_and_declared_dimension_limits_are_checked() {
    // This source has no EOF. The adapter must stop at its fixed byte limit.
    assert!(load_dataset(TooMuch, InputFormat::LibSvm, shape(1, 1), &config()).is_err());
    assert!(InputShape::new(1, 1_000_000_000).is_err());
    assert!(InputShape::new(1_000_000_000, 1).is_err());
    // A large declaration with tiny actual input must not allocate its matrix.
    assert!(
        load_dataset(
            "0\n".as_bytes(),
            InputFormat::LibSvm,
            shape(1_000_000, 65_536),
            &config(),
        )
        .is_err()
    );
}

struct FailedRead;
impl Read for FailedRead {
    fn read(&mut self, _: &mut [u8]) -> io::Result<usize> {
        Err(io::Error::other("test read failure"))
    }
}
struct FailedWrite;
impl io::Write for FailedWrite {
    fn write(&mut self, _: &[u8]) -> io::Result<usize> {
        Err(io::Error::other("test write failure"))
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
#[test]
fn read_and_write_failures_are_returned() {
    assert!(load_dataset(FailedRead, InputFormat::Csv, shape(1, 2), &config()).is_err());
    let model = Model::load(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/models/numeric_l2.txt"),
    )
    .unwrap();
    assert!(
        predict(
            "1,2\n".as_bytes(),
            InputFormat::Csv,
            shape(1, 2),
            &model,
            FailedWrite
        )
        .is_err()
    );
}

fn labeled_text(rows: &[Vec<Option<f64>>], labels: &[f32], format: InputFormat) -> String {
    rows.iter()
        .zip(labels)
        .map(|(row, label)| {
            if format == InputFormat::LibSvm {
                format!(
                    "{label} {}",
                    row.iter()
                        .enumerate()
                        .rev()
                        .filter(|(_, v)| **v != Some(0.0))
                        .map(|(i, v)| format!("{}:{}", i + 1, v.unwrap_or(f64::NAN)))
                        .collect::<Vec<_>>()
                        .join(" ")
                )
            } else {
                let separator = if format == InputFormat::Csv {
                    ","
                } else {
                    "\t"
                };
                format!(
                    "{label}{separator}{}",
                    row.iter()
                        .map(|v| v.map(|v| v.to_string()).unwrap_or_default())
                        .collect::<Vec<_>>()
                        .join(separator)
                )
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
fn refit_adapter_matches_native_routing_and_weighted_library_api() {
    let suite: Value =
        serde_json::from_str(include_str!("fixtures/prediction_operations.json")).unwrap();
    let mut count = 0;
    for case in suite["cases"].as_array().unwrap().iter().filter(|c| {
        let name = c["name"].as_str().unwrap();
        [
            "refit_regression_decay",
            "refit_binary_decay",
            "refit_multiclass_decay",
            "refit_categorical_decay",
            "refit_linear_decay",
        ]
        .iter()
        .any(|prefix| name.starts_with(prefix))
    }) {
        let config =
            Config::from_pairs(case["params"].as_object().unwrap().iter().map(|(k, v)| {
                let value = if let Some(s) = v.as_str() {
                    s.to_owned()
                } else if let Some(a) = v.as_array() {
                    a.iter().map(Value::to_string).collect::<Vec<_>>().join(",")
                } else {
                    v.to_string()
                };
                (k, value)
            }))
            .unwrap();
        let model = Model::load(
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("tests/fixtures")
                .join(case["initial_model"]["model_file"].as_str().unwrap()),
        )
        .unwrap();
        let before = model.to_text().unwrap();
        let rows: Vec<Vec<Option<f64>>> = serde_json::from_value(case["rows"].clone()).unwrap();
        let labels: Vec<f32> = serde_json::from_value(case["labels"].clone()).unwrap();
        let expected_leaves: Vec<Vec<usize>> =
            serde_json::from_value(case["leaf_pred"].clone()).unwrap();
        for format in [InputFormat::Csv, InputFormat::Tsv, InputFormat::LibSvm] {
            let text = labeled_text(&rows, &labels, format);
            let (data, leaves) = load_refit_dataset(
                text.as_bytes(),
                format,
                shape(rows.len(), rows[0].len()),
                &model,
                &config,
            )
            .unwrap();
            assert_eq!(leaves, expected_leaves, "{} {format:?}", case["name"]);
            let data = data
                .with_weights(serde_json::from_value(case["weights"].clone()).unwrap())
                .unwrap();
            let refitted = model
                .refit(
                    &data,
                    &leaves,
                    &config,
                    case["decay_rate"].as_f64().unwrap(),
                )
                .unwrap();
            for (i, row) in rows.iter().enumerate() {
                let row: Vec<f64> = row.iter().map(|v| v.unwrap_or(f64::NAN)).collect();
                let expected = case["final_model"]["transformed"][i].as_array().unwrap();
                for (actual, expected) in refitted.predict_row(&row).unwrap().iter().zip(expected) {
                    let expected = expected.as_f64().unwrap();
                    assert!(
                        (actual - expected).abs() <= 1e-12 + 1e-10 * expected.abs(),
                        "{} {format:?}: {actual} != {expected}",
                        case["name"]
                    );
                }
            }
            assert_eq!(model.to_text().unwrap(), before);
        }
        count += 1;
    }
    assert_eq!(count, 20);
}

#[test]
fn refit_routes_original_f64_before_binning_and_preserves_dense_equivalence() {
    let model = Model::load(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/models/numeric_l2.txt"),
    )
    .unwrap();
    let config = Config::from_pairs([
        ("max_bin", "2"),
        ("min_data_in_bin", "1"),
        ("min_data_in_leaf", "1"),
        ("feature_pre_filter", "false"),
    ])
    .unwrap();
    let rows = vec![
        vec![Some(1.5), Some(0.0)],
        vec![Some(1.5000000000000004), Some(0.0)],
        vec![Some(1e-36), Some(0.0)],
        vec![None, Some(0.0)],
    ];
    let expected: Vec<_> = rows
        .iter()
        .map(|r| {
            model
                .predict_leaf_row(&r.iter().map(|v| v.unwrap_or(f64::NAN)).collect::<Vec<_>>())
                .unwrap()
        })
        .collect();
    assert_ne!(expected[0][1], expected[1][1]);
    for format in [InputFormat::Csv, InputFormat::Tsv, InputFormat::LibSvm] {
        let (data, leaves) = load_refit_dataset(
            labeled_text(&rows, &[0.; 4], format).as_bytes(),
            format,
            shape(4, 2),
            &model,
            &config,
        )
        .unwrap();
        assert_eq!(leaves, expected);
        assert_eq!(data.get_bin(0, 0).unwrap(), data.get_bin(1, 0).unwrap());
    }
}

#[test]
fn refit_reuses_checked_parser_and_checks_limits_before_reading() {
    let model =
        Model::load(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/models/binary.txt"))
            .unwrap();
    let config = Config::from_pairs([("objective", "binary")]).unwrap();
    for text in [
        "",
        "0,1\n",
        "0,1,2,3\n",
        "0,1,2\n1,3,4\n",
        "0.5,1,2\n",
        "NaN,1,2\n",
        "1e40,1,2\n",
        "0,inf,2\n",
    ] {
        assert!(
            load_refit_dataset(
                text.as_bytes(),
                InputFormat::Csv,
                shape(1, 2),
                &model,
                &config
            )
            .is_err(),
            "accepted {text:?}"
        );
    }
    for text in [
        "",
        "0 0:1\n",
        "0 3:1\n",
        "0 1:2 1:3\n",
        "0 qid:1\n",
        "0.5 1:2\n",
    ] {
        assert!(
            load_refit_dataset(
                text.as_bytes(),
                InputFormat::LibSvm,
                shape(1, 2),
                &model,
                &config
            )
            .is_err(),
            "accepted {text:?}"
        );
    }
    assert!(
        load_refit_dataset(FailedRead, InputFormat::Csv, shape(1, 2), &model, &config).is_err()
    );
    assert!(
        load_refit_dataset(TooMuch, InputFormat::LibSvm, shape(1, 2), &model, &config).is_err()
    );
    assert!(
        load_refit_dataset(FailedRead, InputFormat::Csv, shape(1, 3), &model, &config)
            .unwrap_err()
            .to_string()
            .contains("feature count")
    );
    let large = Model::load(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/models/goss_multiclass.txt"),
    )
    .unwrap();
    let error = load_refit_dataset(
        FailedRead,
        InputFormat::LibSvm,
        shape(1_000_000, large.num_features()),
        &large,
        &config,
    )
    .unwrap_err();
    assert!(error.to_string().contains("16777216"));
}
