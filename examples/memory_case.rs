//! Native-only memory comparison. See tools/oracle/measure_memory.py.
//! Both drivers load the same typed little-endian arrays before construction.
use lightgbm_rust::{Config, Dataset, Error, ParallelPredictor, Result, Trainer};
use serde_json::json;
use std::io::{BufReader, Read};
use std::time::Instant;

fn invalid(message: &str) -> Error {
    Error::InvalidInput(message.into())
}
struct Input(BufReader<std::fs::File>);
impl Input {
    fn bytes<const N: usize>(&mut self) -> Result<[u8; N]> {
        let mut bytes = [0; N];
        self.0.read_exact(&mut bytes)?;
        Ok(bytes)
    }
    fn count(&mut self) -> Result<usize> {
        let count = u64::from_le_bytes(self.bytes()?);
        if count > 2_000_000 {
            return Err(invalid("input count exceeds fixed-case limit"));
        }
        usize::try_from(count).map_err(|_| invalid("invalid count"))
    }
    fn f64s(&mut self) -> Result<Vec<f64>> {
        let n = self.count()?;
        (0..n)
            .map(|_| Ok(f64::from_le_bytes(self.bytes()?)))
            .collect()
    }
    fn f32s(&mut self) -> Result<Vec<f32>> {
        let n = self.count()?;
        (0..n)
            .map(|_| Ok(f32::from_le_bytes(self.bytes()?)))
            .collect()
    }
    fn indexes64(&mut self) -> Result<Vec<usize>> {
        let n = self.count()?;
        (0..n)
            .map(|_| {
                usize::try_from(i64::from_le_bytes(self.bytes()?))
                    .map_err(|_| invalid("invalid CSR offset"))
            })
            .collect()
    }
    fn indexes32(&mut self) -> Result<Vec<usize>> {
        let n = self.count()?;
        (0..n)
            .map(|_| {
                usize::try_from(i32::from_le_bytes(self.bytes()?))
                    .map_err(|_| invalid("invalid group/index"))
            })
            .collect()
    }
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 5 {
        return Err(invalid(
            "usage: memory_case INPUT_BIN PARAMS_TEXT THREADS PREDICTIONS_BIN",
        ));
    }
    let mut input = Input(BufReader::new(std::fs::File::open(&args[1])?));
    if &input.bytes::<8>()? != b"LGBMEM01" {
        return Err(invalid("invalid memory input version"));
    }
    let rows = input.count()?;
    let features = input.count()?;
    let valid_rows = input.count()?;
    let rounds = input.count()?;
    let sparse = input.count()? == 1;
    if (rows, features, valid_rows, rounds) != (8192, 16, 2048, 50) {
        return Err(invalid("not a fixed benchmark case"));
    }
    let train = input.f64s()?;
    let valid = input.f64s()?;
    let labels = input.f32s()?;
    let valid_labels = input.f32s()?;
    let weights = input.f32s()?;
    let groups = input.indexes32()?;
    let train_ptr = input.indexes64()?;
    let train_idx = input.indexes32()?;
    let train_data = input.f64s()?;
    let valid_ptr = input.indexes64()?;
    let valid_idx = input.indexes32()?;
    let valid_data = input.f64s()?;
    if input.0.read(&mut [0])? != 0 || labels.len() != rows || valid_labels.len() != valid_rows {
        return Err(invalid("invalid input shape or trailing data"));
    }
    drop(input);
    let params = std::fs::read_to_string(&args[2])?;
    let pairs = params
        .split_ascii_whitespace()
        .map(|p| {
            p.split_once('=')
                .ok_or_else(|| invalid("invalid parameter pair"))
        })
        .collect::<Result<Vec<_>>>()?;
    let mut config = Config::from_pairs(pairs)?;
    config.num_iterations = rounds;
    config.num_threads = args[3]
        .parse()
        .map_err(|_| invalid("invalid thread count"))?;
    config.validate()?;
    let predictor = ParallelPredictor::new(config.num_threads)?;
    let mut construction = Vec::new();
    let mut training = Vec::new();
    let mut inference = Vec::new();
    let mut serialization = Vec::new();
    let mut predictions = Vec::new();
    for repetition in 0..6 {
        let start = Instant::now();
        let dataset = if sparse {
            Dataset::from_csr(
                &train_ptr,
                &train_idx,
                &train_data,
                rows,
                features,
                labels.clone(),
                &config,
            )?
        } else {
            Dataset::from_dense(&train, rows, features, labels.clone(), &config)?
        };
        let dataset = if weights.is_empty() {
            dataset
        } else {
            dataset.with_weights(weights.clone())?
        };
        let dataset = if groups.is_empty() {
            dataset
        } else {
            dataset.with_groups(groups.clone())?
        };
        let built = Instant::now();
        // fit includes the product's training metrics/history, even with metric=None.
        let result = Trainer::new(config.clone())?.fit(&dataset)?;
        let trained = Instant::now();
        let text = result.model.to_text()?;
        std::hint::black_box(&text);
        let saved = Instant::now();
        let predicted = if sparse {
            predictor.predict_csr(&result.model, &valid_ptr, &valid_idx, &valid_data)?
        } else {
            predictor.predict_dense(&result.model, &valid, valid_rows, features)?
        };
        let done = Instant::now();
        if repetition > 0 {
            construction.push((built - start).as_secs_f64());
            training.push((trained - built).as_secs_f64());
            serialization.push((saved - trained).as_secs_f64());
            inference.push((done - saved).as_secs_f64());
        }
        // Drop the previous prediction before constructing the next dataset.
        if repetition == 5 {
            predictions = predicted;
        }
    }
    let flat: Vec<f64> = predictions.into_iter().flatten().collect();
    let mut output = std::io::BufWriter::new(std::fs::File::create(&args[4])?);
    for value in &flat {
        std::io::Write::write_all(&mut output, &value.to_le_bytes())?;
    }
    println!(
        "{}",
        json!({"warmup_runs":1,"measured_runs":5,"threads":config.num_threads,
        "dataset_construction_seconds":construction,"training_seconds":training,
        "serialization_seconds":serialization,"inference_seconds":inference,
        "prediction_count":flat.len()})
    );
    Ok(())
}
