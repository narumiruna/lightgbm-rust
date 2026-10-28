//! CLI checks use the saved native oracle and the actual Cargo binary.
use lightgbm_rust::{Config, Dataset, Model, Trainer};
use serde::Deserialize;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicUsize, Ordering};

static NEXT: AtomicUsize = AtomicUsize::new(0);
struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "lightgbm-cli-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
    fn path(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
#[derive(Deserialize)]
struct Suite {
    cases: Vec<Case>,
}
#[derive(Deserialize)]
struct Case {
    name: String,
    model_file: String,
    rows: Vec<Vec<Option<f64>>>,
    transformed: Vec<Vec<f64>>,
    training: Training,
}
#[derive(Deserialize)]
struct Training {
    rows: Vec<Vec<Option<f64>>>,
    labels: Vec<f32>,
}
fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}
fn cases() -> Vec<Case> {
    serde_json::from_str::<Suite>(include_str!("fixtures/prediction.json"))
        .unwrap()
        .cases
}
fn command(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_lightgbm-rust"))
        .args(args)
        .output()
        .unwrap()
}
fn ok(output: &Output) {
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}
fn compare(output: &[u8], expected: &[Vec<f64>]) {
    let text = std::str::from_utf8(output).unwrap();
    let rows: Vec<Vec<f64>> = text
        .lines()
        .map(|line| line.split('\t').map(|v| v.parse().unwrap()).collect())
        .collect();
    assert_eq!(rows.len(), expected.len());
    for (actual, expected) in rows.iter().zip(expected) {
        assert_eq!(actual.len(), expected.len());
        for (&a, &b) in actual.iter().zip(expected) {
            assert!(
                a.is_finite() && (a - b).abs() <= 1e-12 + 1e-10 * b.abs(),
                "{a} != {b}"
            );
        }
    }
}
fn row_text(row: &[Option<f64>], delimiter: &str) -> String {
    row.iter()
        .map(|v| v.map(|v| v.to_string()).unwrap_or_else(|| "\"\"".into()))
        .collect::<Vec<_>>()
        .join(delimiter)
}

#[test]
fn imported_native_models_predict_all_formats_and_shapes() {
    let dir = Temp::new();
    for case in cases() {
        for format in ["csv", "tsv", "libsvm"] {
            let text = case
                .rows
                .iter()
                .map(|row| {
                    if format == "libsvm" {
                        // A label is required in LibSVM, but it is not a prediction.
                        format!(
                            "123 {}",
                            row.iter()
                                .enumerate()
                                .rev()
                                .map(|(i, value)| {
                                    format!(
                                        "{}:{}",
                                        i + 1,
                                        value
                                            .map(|v| v.to_string())
                                            .unwrap_or_else(|| "NaN".into())
                                    )
                                })
                                .collect::<Vec<_>>()
                                .join(" ")
                        )
                    } else {
                        row_text(row, if format == "csv" { "," } else { "\t" })
                    }
                })
                .collect::<Vec<_>>()
                .join("\n");
            let input = dir.path("input");
            fs::write(&input, text).unwrap();
            let model = fixtures().join(&case.model_file);
            let out = command(&[
                "predict",
                "--input",
                input.to_str().unwrap(),
                "--format",
                format,
                "--rows",
                &case.rows.len().to_string(),
                "--features",
                &case.rows[0].len().to_string(),
                "--model",
                model.to_str().unwrap(),
            ]);
            ok(&out);
            compare(&out.stdout, &case.transformed);
        }
    }
}

#[test]
fn train_save_refit_load_predict_matches_library_for_three_objectives_and_formats() {
    let dir = Temp::new();
    for case in cases()
        .into_iter()
        .filter(|c| ["numeric_l2", "binary", "multiclass"].contains(&c.name.as_str()))
    {
        let objective = match case.name.as_str() {
            "binary" => "binary",
            "multiclass" => "multiclass",
            _ => "regression",
        };
        let pairs = [
            ("objective", objective),
            (
                "num_class",
                if objective == "multiclass" { "3" } else { "1" },
            ),
            ("num_iterations", "3"),
            ("num_leaves", "3"),
            ("learning_rate", "0.25"),
            ("min_data_in_leaf", "1"),
            ("min_data_in_bin", "1"),
            ("max_bin", "15"),
            ("feature_pre_filter", "false"),
        ];
        let config = Config::from_pairs(pairs).unwrap();
        let flat: Vec<f64> = case
            .training
            .rows
            .iter()
            .flatten()
            .map(|v| v.unwrap_or(f64::NAN))
            .collect();
        let dataset = Dataset::from_dense(
            &flat,
            case.training.rows.len(),
            case.training.rows[0].len(),
            case.training.labels.clone(),
            &config,
        )
        .unwrap();
        let expected = Trainer::new(config.clone())
            .unwrap()
            .fit(&dataset)
            .unwrap()
            .model;
        let leaves = expected
            .predict_leaf_dense(&flat, dataset.nrows(), dataset.ncols())
            .unwrap();
        let refitted = expected.refit(&dataset, &leaves, &config, 0.5).unwrap();
        for format in ["csv", "tsv", "libsvm"] {
            let input = dir.path("train");
            let model = dir.path(&format!("{}-{format}.txt", case.name));
            let text = case
                .training
                .rows
                .iter()
                .zip(&case.training.labels)
                .map(|(row, label)| {
                    if format == "libsvm" {
                        format!(
                            "{label} {}",
                            row.iter()
                                .enumerate()
                                .rev()
                                .map(|(i, v)| format!("{}:{}", i + 1, v.unwrap_or(f64::NAN)))
                                .collect::<Vec<_>>()
                                .join(" ")
                        )
                    } else {
                        let delimiter = if format == "csv" { "," } else { "\t" };
                        format!(
                            "{label}{delimiter}{}",
                            row.iter()
                                .map(|v| v.map(|v| v.to_string()).unwrap_or_default())
                                .collect::<Vec<_>>()
                                .join(delimiter)
                        )
                    }
                })
                .collect::<Vec<_>>()
                .join("\n");
            fs::write(&input, text).unwrap();
            let mut args = vec![
                "train".to_owned(),
                "--input".into(),
                input.to_str().unwrap().into(),
                "--format".into(),
                format.into(),
                "--rows".into(),
                dataset.nrows().to_string(),
                "--features".into(),
                dataset.ncols().to_string(),
                "--model".into(),
                model.to_str().unwrap().into(),
            ];
            args.extend(pairs.iter().map(|(k, v)| format!("{k}={v}")));
            ok(&command(
                &args.iter().map(String::as_str).collect::<Vec<_>>(),
            ));
            let loaded = Model::load(&model).unwrap();
            let rows = case
                .rows
                .iter()
                .map(|row| {
                    row.iter()
                        .map(|v| v.unwrap_or(f64::NAN))
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            for row in &rows {
                assert_eq!(
                    loaded.predict_dense(row, 1, row.len()).unwrap(),
                    expected.predict_dense(row, 1, row.len()).unwrap()
                );
            }
            let original_bytes = fs::read(&model).unwrap();
            let new_model = dir.path(&format!("{}-{format}-refit.txt", case.name));
            let mut refit_args = args.clone();
            refit_args[0] = "refit".into();
            refit_args.extend([
                "--output".into(),
                new_model.to_str().unwrap().into(),
                "--decay-rate".into(),
                "0.5".into(),
                "app=regression".into(),
            ]);
            ok(&command(
                &refit_args.iter().map(String::as_str).collect::<Vec<_>>(),
            ));
            assert_eq!(fs::read(&model).unwrap(), original_bytes);
            let reloaded = Model::load(&new_model).unwrap();
            for row in &rows {
                assert_eq!(
                    reloaded.predict_row(row).unwrap(),
                    refitted.predict_row(row).unwrap()
                );
            }
            let predict = dir.path("predict");
            fs::write(
                &predict,
                case.rows
                    .iter()
                    .map(|r| row_text(r, ","))
                    .collect::<Vec<_>>()
                    .join("\n"),
            )
            .unwrap();
            let out = command(&[
                "predict",
                "--input",
                predict.to_str().unwrap(),
                "--format",
                "csv",
                "--rows",
                &rows.len().to_string(),
                "--features",
                &dataset.ncols().to_string(),
                "--model",
                new_model.to_str().unwrap(),
            ]);
            ok(&out);
            compare(
                &out.stdout,
                &rows
                    .iter()
                    .map(|r| refitted.predict_dense(r, 1, r.len()).unwrap().remove(0))
                    .collect::<Vec<_>>(),
            );
            let before = fs::read(&model).unwrap();
            assert!(
                !command(&args.iter().map(String::as_str).collect::<Vec<_>>())
                    .status
                    .success()
            );
            assert_eq!(fs::read(&model).unwrap(), before);
        }
    }
}

#[test]
fn malformed_arguments_data_and_refit_fail_without_clobber() {
    let dir = Temp::new();
    let input = dir.path("input");
    let output = dir.path("output");
    let model = fixtures().join("models/numeric_l2.txt");
    fs::write(&input, "1,2\n").unwrap();
    let base = vec![
        "predict",
        "--input",
        input.to_str().unwrap(),
        "--format",
        "csv",
        "--rows",
        "1",
        "--features",
        "2",
        "--model",
        model.to_str().unwrap(),
    ];
    for extra in [
        vec!["--header"],
        vec!["--weight-column", "0"],
        vec!["--rows", "1"],
        vec!["objective=regression"],
        vec!["--output"],
    ] {
        let mut args = base.clone();
        args.extend(extra);
        let out = command(&args);
        assert!(!out.status.success());
        assert!(!out.stderr.is_empty());
    }
    for text in ["header,name\n", "1,2,3\n", "1,inf\n", "1,2\n3,4\n", ""] {
        fs::write(&input, text).unwrap();
        let mut args = base.clone();
        args.extend(["--output", output.to_str().unwrap()]);
        let out = command(&args);
        assert!(!out.status.success(), "accepted {text:?}");
        assert!(!out.stderr.is_empty());
        assert!(
            !output.exists(),
            "failed prediction published partial output"
        );
    }
    for (option, value) in [
        ("--rows", "0"),
        ("--rows", "2"),
        ("--features", "3"),
        ("--features", "184467440737095516160"),
        ("--format", "parquet"),
    ] {
        fs::write(&input, "1,2\n").unwrap();
        let mut args = base.clone();
        let position = args.iter().position(|arg| *arg == option).unwrap();
        args[position + 1] = value;
        let out = command(&args);
        assert!(!out.status.success(), "accepted {option} {value}");
        assert!(!out.stderr.is_empty());
    }
    for text in [
        "0 0:2\n",
        "0 1:2 1:3\n",
        "NaN 1:2\n",
        "0 3:2\n",
        "0 1:2:3\n",
    ] {
        fs::write(&input, text).unwrap();
        let mut args = base.clone();
        args[4] = "libsvm";
        let out = command(&args);
        assert!(!out.status.success(), "accepted LibSVM {text:?}");
        assert!(!out.stderr.is_empty());
    }
    fs::write(&input, "1,2\n").unwrap();
    let mut args = base.clone();
    args.extend(["--output", output.to_str().unwrap()]);
    ok(&command(&args));
    let original = fs::read(&output).unwrap();
    assert!(!command(&args).status.success());
    assert_eq!(fs::read(&output).unwrap(), original);
    for source in [&input, &model] {
        let before = fs::read(source).unwrap();
        let mut args = base.clone();
        args.extend(["--output", source.to_str().unwrap()]);
        assert!(!command(&args).status.success());
        assert_eq!(fs::read(source).unwrap(), before);
    }
    let out = command(&["refit"]);
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("missing required option"));
    assert!(String::from_utf8_lossy(&command(&["--help"]).stdout).contains("--features"));
}

#[test]
fn malformed_training_parameters_and_labels_fail() {
    let dir = Temp::new();
    let input = dir.path("input");
    let model = dir.path("model");
    for (text, parameter) in [
        ("0,1,2\n", "unknown=1"),
        ("0,1,2\n", "learning_rate=NaN"),
        ("NaN,1,2\n", "objective=regression"),
        ("1e40,1,2\n", "objective=regression"),
        ("0.2,1,2\n", "objective=binary"),
        ("-1,1,2\n", "objective=binary"),
        ("0,1,2\n", "categorical_feature=2"),
    ] {
        fs::write(&input, text).unwrap();
        let out = command(&[
            "train",
            "--input",
            input.to_str().unwrap(),
            "--format",
            "csv",
            "--rows",
            "1",
            "--features",
            "2",
            "--model",
            model.to_str().unwrap(),
            parameter,
        ]);
        assert!(!out.status.success(), "accepted {text:?} {parameter}");
        assert!(!out.stderr.is_empty());
        assert!(!model.exists());
    }
}

#[test]
fn refit_errors_preserve_original_model_dataset_and_existing_destination() {
    let dir = Temp::new();
    let input = dir.path("input");
    let original = dir.path("original.txt");
    let output = dir.path("new.txt");
    let source = fs::read(fixtures().join("models/binary.txt")).unwrap();
    fs::write(&original, &source).unwrap();
    let valid = "0,-3,0\n1,3,1\n";
    fs::write(&input, valid).unwrap();
    let base = vec![
        "refit",
        "--input",
        input.to_str().unwrap(),
        "--format",
        "csv",
        "--rows",
        "2",
        "--features",
        "2",
        "--model",
        original.to_str().unwrap(),
        "--output",
        output.to_str().unwrap(),
        "objective=binary",
        "min_data_in_leaf=1",
        "min_data_in_bin=1",
        "feature_pre_filter=false",
    ];
    for extra in [
        vec!["--unknown", "1"],
        vec!["--weight-column", "0"],
        vec!["--group-column", "0"],
        vec!["--rows", "2"],
        vec!["--output", "x"],
        vec!["objective=binary"],
        vec!["lambda_l2=1", "lambda_l2=2"],
        vec!["unknown=1"],
        vec!["lambda_l2=NaN"],
        vec!["sigmoid=2"],
        vec!["num_class=3"],
        vec!["reg_sqrt=true"],
        vec!["categorical_feature=2"],
        vec!["--decay-rate", "-0.1"],
        vec!["--decay-rate", "1.1"],
        vec!["--decay-rate", "NaN"],
        vec!["--decay-rate", "inf"],
        vec!["--decay-rate", "value"],
        vec!["--decay-rate", "0", "--decay-rate", "1"],
    ] {
        let mut args = base.clone();
        args.extend(extra);
        let result = command(&args);
        assert!(!result.status.success(), "accepted {args:?}");
        assert!(!result.stderr.is_empty());
        assert!(!output.exists());
        assert_eq!(fs::read(&original).unwrap(), source);
        assert_eq!(fs::read_to_string(&input).unwrap(), valid);
    }
    for text in [
        "",
        "0,-3\n1,3,1\n",
        "0,-3,0\n",
        "label,a,b\n0,1,2\n",
        "0,-3,0\n0.5,3,1\n",
        "0,-3,0\nNaN,3,1\n",
        "0,-3,0\n1,inf,1\n",
        "0,-3,0\n1,3,1\n0,0,0\n",
    ] {
        fs::write(&input, text).unwrap();
        assert!(!command(&base).status.success(), "accepted {text:?}");
        assert!(!output.exists());
        assert_eq!(fs::read(&original).unwrap(), source);
        assert_eq!(fs::read_to_string(&input).unwrap(), text);
    }
    fs::write(&input, valid).unwrap();
    for (flag, value) in [
        ("--rows", "0"),
        ("--features", "0"),
        ("--rows", "3"),
        ("--features", "3"),
        ("--format", "parquet"),
    ] {
        let mut args = base.clone();
        let index = args.iter().position(|arg| *arg == flag).unwrap();
        args[index + 1] = value;
        assert!(!command(&args).status.success());
        assert!(!output.exists());
    }
    let mut no_objective = base.clone();
    no_objective.retain(|arg| !arg.starts_with("objective="));
    assert!(
        String::from_utf8_lossy(&command(&no_objective).stderr)
            .contains("explicit matching objective")
    );
    for objective in ["objective=regression", "objective=lambdarank"] {
        let mut args = base.clone();
        args[13] = objective;
        let result = command(&args);
        assert!(!result.status.success());
        if objective.contains("lambdarank") {
            assert!(String::from_utf8_lossy(&result.stderr).contains("Rust Dataset API"));
        }
    }
    // The destination must be new, even when it aliases a source path.
    fs::write(&output, "existing destination").unwrap();
    for path in [&output, &input, &original] {
        let mut args = base.clone();
        args[12] = path.to_str().unwrap();
        assert!(!command(&args).status.success());
        assert_eq!(fs::read(&original).unwrap(), source);
        assert_eq!(fs::read_to_string(&input).unwrap(), valid);
        assert_eq!(fs::read_to_string(&output).unwrap(), "existing destination");
    }
    fs::write(&input, "0,-3,0\n0.5,3,1\n").unwrap();
    assert!(!command(&base).status.success());
    assert_eq!(fs::read_to_string(&output).unwrap(), "existing destination");
    assert_eq!(fs::read(&original).unwrap(), source);
    fs::write(&input, valid).unwrap();
    fs::remove_file(&output).unwrap();
    ok(&command(&base)); // default decay = 0.9
    let config = Config::from_pairs([
        ("objective", "binary"),
        ("min_data_in_leaf", "1"),
        ("min_data_in_bin", "1"),
        ("feature_pre_filter", "false"),
    ])
    .unwrap();
    let old = Model::load(&original).unwrap();
    let rows = [-3., 0., 3., 1.];
    let data = Dataset::from_dense(&rows, 2, 2, vec![0., 1.], &config).unwrap();
    let leaves = old.predict_leaf_dense(&rows, 2, 2).unwrap();
    let expected = old.refit(&data, &leaves, &config, 0.9).unwrap();
    assert_eq!(
        Model::load(&output).unwrap().to_text().unwrap(),
        expected.to_text().unwrap()
    );
    let saved = fs::read(&output).unwrap();
    assert!(!command(&base).status.success());
    assert_eq!(fs::read(&output).unwrap(), saved);
    let names: Vec<_> = fs::read_dir(&dir.0)
        .unwrap()
        .map(|e| e.unwrap().file_name())
        .collect();
    assert!(!names.iter().any(|n| n.to_string_lossy().ends_with(".tmp")));
}

#[test]
fn imported_native_model_refit_decay_one_preserves_predictions_all_formats() {
    let dir = Temp::new();
    for case in cases().into_iter().filter(|c| {
        [
            "numeric_l2",
            "binary",
            "multiclass",
            "binary_sigmoid",
            "regression_sqrt",
        ]
        .contains(&c.name.as_str())
    }) {
        let original = fixtures().join(&case.model_file);
        let original_bytes = fs::read(&original).unwrap();
        let objective = match case.name.as_str() {
            "binary" | "binary_sigmoid" => "binary",
            "multiclass" => "multiclass",
            _ => "regression",
        };
        for format in ["csv", "tsv", "libsvm"] {
            let input = dir.path("input");
            let output = dir.path(&format!("{}-{format}.txt", case.name));
            let text = case
                .training
                .rows
                .iter()
                .zip(&case.training.labels)
                .map(|(row, label)| {
                    if format == "libsvm" {
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
                        let sep = if format == "csv" { "," } else { "\t" };
                        format!("{label}{sep}{}", row_text(row, sep))
                    }
                })
                .collect::<Vec<_>>()
                .join("\n");
            fs::write(&input, text).unwrap();
            let result = command(&[
                "refit",
                "--input",
                input.to_str().unwrap(),
                "--format",
                format,
                "--rows",
                &case.training.rows.len().to_string(),
                "--features",
                &case.rows[0].len().to_string(),
                "--model",
                original.to_str().unwrap(),
                "--output",
                output.to_str().unwrap(),
                "--decay-rate",
                "1",
                &format!("loss={objective}"),
                if objective == "multiclass" {
                    "num_classes=3"
                } else {
                    "num_class=1"
                },
                if case.name == "binary_sigmoid" {
                    "sigmoid=2"
                } else {
                    "sigmoid=1"
                },
                if case.name == "regression_sqrt" {
                    "reg_sqrt=true"
                } else {
                    "reg_sqrt=false"
                },
            ]);
            ok(&result);
            let loaded = Model::load(&output).unwrap();
            let predictions: Vec<_> = case
                .rows
                .iter()
                .map(|r| {
                    loaded
                        .predict_row(&r.iter().map(|v| v.unwrap_or(f64::NAN)).collect::<Vec<_>>())
                        .unwrap()
                })
                .collect();
            let text = predictions
                .iter()
                .map(|r| r.iter().map(f64::to_string).collect::<Vec<_>>().join("\t"))
                .collect::<Vec<_>>()
                .join("\n");
            compare(text.as_bytes(), &case.transformed);
            assert_eq!(fs::read(&original).unwrap(), original_bytes);
        }
    }
}
