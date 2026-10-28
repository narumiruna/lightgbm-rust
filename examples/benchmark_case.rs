//! Compare a supported fixed case exported by tools/oracle/benchmark.py.
//! Input parsing is outside all measured construction/training/prediction spans.
use lightgbm_rust::{Config, Dataset, Error, ParallelPredictor, Result, Trainer};
use serde::Deserialize;
use serde_json::json;
use std::io::Read;
use std::time::Instant;

#[derive(Deserialize)]
struct Csr {
    indptr: Vec<usize>,
    indices: Vec<usize>,
    values: Vec<f64>,
}
#[derive(Deserialize)]
struct Input {
    schema_version: u32,
    upstream_commit: String,
    name: String,
    rows: usize,
    features: usize,
    validation_rows: usize,
    num_boost_round: usize,
    train_values: Vec<f64>,
    train_labels: Vec<f32>,
    validation_values: Vec<f64>,
    validation_labels: Vec<f32>,
    train_csr: Option<Csr>,
    validation_csr: Option<Csr>,
    params: serde_json::Value,
    reference_expected_quality: f64,
    reference_median_training_seconds: f64,
    reference_median_inference_seconds: f64,
}
fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    sorted[sorted.len() / 2]
}
// Source-equivalent tools/oracle/benchmark.py::quality. Stable query ordering
// retains input-row order for tied scores, as NumPy argsort(kind="stable").
fn quality(name: &str, labels: &[f32], predictions: &[Vec<f64>]) -> f64 {
    if name == "ranking" {
        let mut sum = 0.0;
        for (labels, scores) in labels.chunks_exact(16).zip(predictions.chunks_exact(16)) {
            let mut ideal = labels.to_vec();
            ideal.sort_by(|a, b| b.total_cmp(a));
            let mut order: Vec<_> = (0..16).collect();
            order.sort_by(|&a, &b| scores[b][0].total_cmp(&scores[a][0]));
            let dcg = |values: &[f32]| {
                values
                    .iter()
                    .take(5)
                    .enumerate()
                    .map(|(i, &y)| (2.0_f64.powf(f64::from(y)) - 1.0) / ((i + 2) as f64).log2())
                    .sum::<f64>()
            };
            let selected: Vec<_> = order.iter().map(|&i| labels[i]).collect();
            let denominator = dcg(&ideal);
            sum += if denominator == 0.0 {
                1.0
            } else {
                dcg(&selected) / denominator
            };
        }
        return sum / (labels.len() / 16) as f64;
    }
    let mean = predictions
        .iter()
        .zip(labels)
        .map(|(p, &y)| match name {
            "categorical" => {
                let p = p[0].clamp(1e-15, 1.0 - 1e-15);
                -f64::from(y) * p.ln() - (1.0 - f64::from(y)) * (1.0 - p).ln()
            }
            "multiclass" => -p[y as usize].max(1e-15).ln(),
            _ => (p[0] - f64::from(y)).powi(2),
        })
        .sum::<f64>()
        / predictions.len() as f64;
    if name == "dense" || name == "sparse" {
        mean.sqrt()
    } else {
        mean
    }
}

fn main() -> Result<()> {
    let path = std::env::args().nth(1).ok_or_else(|| {
        Error::InvalidInput("usage: benchmark_case INPUT_JSON [NUM_THREADS]".into())
    })?;
    let mut text = String::new();
    std::fs::File::open(&path)?
        .take(64 * 1024 * 1024 + 1)
        .read_to_string(&mut text)?;
    if text.len() > 64 * 1024 * 1024 {
        return Err(Error::InvalidInput("benchmark input exceeds 64 MiB".into()));
    }
    let input: Input =
        serde_json::from_str(&text).map_err(|e| Error::InvalidInput(e.to_string()))?;
    if input.schema_version != 1
        || input.upstream_commit != "5d84f3f222b41409a6983d2ac90513ad9702fa9e"
        || !["dense", "sparse", "categorical", "multiclass", "ranking"]
            .contains(&input.name.as_str())
        || input.rows != 8192
        || input.validation_rows != 2048
        || input.features != 16
        || input.num_boost_round != 50
        || input.params["max_bin"] != 63
        || input.params["num_leaves"] != 15
    {
        return Err(Error::InvalidInput(
            "input is not a supported fixed benchmark contract".into(),
        ));
    }
    let expected_objective = match input.name.as_str() {
        "categorical" => "binary",
        "multiclass" => "multiclass",
        "ranking" => "lambdarank",
        _ => "regression",
    };
    if input.params["objective"] != expected_objective {
        return Err(Error::InvalidInput(
            "benchmark objective does not match case".into(),
        ));
    }
    let params = input
        .params
        .as_object()
        .ok_or_else(|| Error::InvalidInput("params must be an object".into()))?;
    let pairs: Vec<_> = params
        .iter()
        .map(|(key, value)| {
            (
                key.as_str(),
                value
                    .as_str()
                    .map(str::to_owned)
                    .unwrap_or_else(|| value.to_string()),
            )
        })
        .collect();
    let mut config = Config::from_pairs(pairs)?;
    config.num_iterations = input.num_boost_round;
    if input.name == "categorical" {
        config.categorical_features = vec![0];
    }
    // Optional supplemental parallel runs do not change the one-thread contract.
    if let Some(threads) = std::env::args().nth(2) {
        config.num_threads = threads
            .parse()
            .map_err(|_| Error::InvalidInput("invalid thread count".into()))?;
    }
    config.validate()?;
    let predictor = ParallelPredictor::new(config.num_threads)?;
    let mut construction = Vec::new();
    let mut training = Vec::new();
    let mut inference = Vec::new();
    let mut qualities = Vec::new();
    let total = Instant::now();
    for repetition in 0..6 {
        let start = Instant::now();
        let dataset = if let Some(csr) = &input.train_csr {
            Dataset::from_csr(
                &csr.indptr,
                &csr.indices,
                &csr.values,
                input.rows,
                input.features,
                input.train_labels.clone(),
                &config,
            )?
        } else {
            Dataset::from_dense(
                &input.train_values,
                input.rows,
                input.features,
                input.train_labels.clone(),
                &config,
            )?
        };
        let dataset = if input.name == "ranking" {
            dataset.with_groups(vec![16; input.rows / 16])?
        } else {
            dataset
        };
        let built = Instant::now();
        let result = Trainer::new(config.clone())?.fit(&dataset)?;
        let trained = Instant::now();
        let predicted = if let Some(csr) = &input.validation_csr {
            if csr.indptr.len() != input.validation_rows + 1
                || csr.indices.len() != csr.values.len()
                || csr.indptr.first() != Some(&0)
                || csr.indptr.last() != Some(&csr.values.len())
                || csr.indptr.windows(2).any(|p| p[0] > p[1])
            {
                return Err(Error::InvalidInput("invalid benchmark CSR shape".into()));
            }
            predictor.predict_csr(&result.model, &csr.indptr, &csr.indices, &csr.values)?
        } else {
            predictor.predict_dense(
                &result.model,
                &input.validation_values,
                input.validation_rows,
                input.features,
            )?
        };
        let predicted_at = Instant::now();
        if predicted.len() != input.validation_labels.len() {
            return Err(Error::InvalidInput(
                "validation labels do not match predictions".into(),
            ));
        }
        let quality = quality(&input.name, &input.validation_labels, &predicted);
        if repetition > 0 {
            construction.push((built - start).as_secs_f64());
            training.push((trained - built).as_secs_f64());
            inference.push((predicted_at - trained).as_secs_f64());
            qualities.push(quality);
        }
        if total.elapsed().as_secs() > 300 {
            return Err(Error::InvalidInput("case exceeded 300 second limit".into()));
        }
    }
    let train_ratio = median(&training) / input.reference_median_training_seconds;
    let inference_ratio = median(&inference) / input.reference_median_inference_seconds;
    let bound = match input.name.as_str() {
        "dense" | "multiclass" => 0.45,
        "sparse" => 0.30,
        "categorical" => 0.15,
        _ => 0.90,
    };
    let ranking = input.name == "ranking";
    let delta = if ranking { 0.01 } else { 0.02 };
    let quality_pass = qualities.iter().all(|&q| {
        q.is_finite()
            && if ranking {
                q >= bound && q >= input.reference_expected_quality - delta
            } else {
                q <= bound && q <= input.reference_expected_quality + delta
            }
    }) && qualities.windows(2).all(|p| p[0] == p[1]);
    let matched_threads = predictor.num_threads() == 1;
    let report = json!({
        "schema_version":1,"name":input.name,"input":path,"upstream_commit":input.upstream_commit,
        "build":"cargo --release, Rust 1.92.0", "rows":input.rows,"features":input.features,"validation_rows":input.validation_rows,
        "warmup_runs":1,"measured_runs":5,"num_boost_round":50,"threads":predictor.num_threads(),
        "matched_reference_threads":matched_threads,
        "dataset_construction_seconds":construction,"training_seconds":training,"inference_seconds":inference,
        "quality":qualities,"reference_quality":input.reference_expected_quality,"absolute_quality_bound":bound,"reference_quality_delta_bound":delta,
        "reference_median_training_seconds":input.reference_median_training_seconds,
        "reference_median_inference_seconds":input.reference_median_inference_seconds,
        "training_ratio":train_ratio,"inference_ratio":inference_ratio,"quality_pass":quality_pass,
        "training_budget_pass":matched_threads.then_some(train_ratio <= 1.5),
        "inference_budget_pass":matched_threads.then_some(inference_ratio <= 1.25),
        "memory_budget":"unverified: matched native-only peak RSS evidence is required",
        "all_cpu_cases":"this report covers only the named case; matched memory remains open"
    });
    println!(
        "{}",
        serde_json::to_string_pretty(&report).map_err(|e| Error::InvalidInput(e.to_string()))?
    );
    if !quality_pass || (matched_threads && (train_ratio > 1.5 || inference_ratio > 1.25)) {
        return Err(Error::InvalidInput(
            "one or more declared benchmark gates failed; see JSON report".into(),
        ));
    }
    Ok(())
}
