//! Numeric CSV, TSV, and LibSVM adapters for the checked Rust API.
//!
//! CSV/TSV have no header. Training and refit rows contain a label followed by exactly
//! `features` values. Prediction rows contain features only. Quoted fields use
//! the `csv` crate rules, including ignored empty lines. Use `""` for a
//! one-field missing row. Blank feature fields and `NaN` mean missing values.
//! Infinity and nonnumeric fields are errors. Labels convert to finite `f32`;
//! binary training and refit require exactly 0 or 1 before conversion.
//!
//! LibSVM rows always start with a finite `f32` label, including prediction
//! input. Prediction discards that label. Feature indexes are one-based;
//! unspecified values are zero. Unsorted indexes are accepted, duplicates are
//! errors. Comments, query IDs, and blank lines are not supported.
//!
//! Callers must supply exact row and feature counts. Inputs are limited to
//! 64 MiB, 1,000,000 rows, and 65,536 features. Training and refit store at most
//! 8,388,608 supplied feature values. No matrix is allocated from declared dimensions.
//! Prediction processes one row at a time after the bounded input read.
//!
//! There are no weight columns, string-category encoders, or pandas adapters.
//! An extra column is an error, not a weight or an ignored field. Callers must
//! encode categories with stable numeric indexes used by the imported model.

use crate::{Config, Dataset, Error, Model, Objective, Result};
use std::fs::OpenOptions;
use std::io::{BufWriter, Read, Write};
use std::path::Path;
use std::str::FromStr;
use std::sync::atomic::{AtomicUsize, Ordering};

const MAX_INPUT_BYTES: usize = 64 * 1024 * 1024;
const MAX_ROWS: usize = 1_000_000;
const MAX_FEATURES: usize = 65_536;
const MAX_TRAIN_VALUES: usize = 8 * 1024 * 1024;
const MAX_REFIT_ENTRIES: usize = 16 * 1024 * 1024;
static TEMP_ID: AtomicUsize = AtomicUsize::new(0);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InputFormat {
    Csv,
    Tsv,
    LibSvm,
}

impl FromStr for InputFormat {
    type Err = Error;
    fn from_str(value: &str) -> Result<Self> {
        match value {
            "csv" => Ok(Self::Csv),
            "tsv" => Ok(Self::Tsv),
            "libsvm" => Ok(Self::LibSvm),
            _ => Err(Error::Unsupported(format!(
                "input format {value:?}; expected csv, tsv, or libsvm"
            ))),
        }
    }
}

/// Explicit input dimensions, checked before any dimension-based allocation.
#[derive(Clone, Copy, Debug)]
pub struct InputShape {
    rows: usize,
    features: usize,
}

impl InputShape {
    pub fn new(rows: usize, features: usize) -> Result<Self> {
        if !(1..=MAX_ROWS).contains(&rows) || !(1..=MAX_FEATURES).contains(&features) {
            return Err(invalid(
                "shape requires 1..=1000000 rows and 1..=65536 features",
            ));
        }
        rows.checked_mul(features)
            .and_then(|n| n.checked_mul(std::mem::size_of::<f64>()))
            .filter(|&n| n <= isize::MAX as usize)
            .ok_or_else(|| invalid("input shape overflows allocation limits"))?;
        Ok(Self { rows, features })
    }
}

enum Features {
    Dense(Vec<f64>),
    Sparse(Vec<(usize, f64)>),
}

struct Row {
    label: Option<f32>,
    features: Features,
}

/// Build a training dataset. Parameters and objective label domains are checked.
/// Use `Dataset::with_weights` separately for library-supplied weights.
pub fn load_dataset(
    reader: impl Read,
    format: InputFormat,
    shape: InputShape,
    config: &Config,
) -> Result<Dataset> {
    load_labeled_dataset(reader, format, shape, config, None).map(|(data, _)| data)
}

/// Load labeled refit rows and route them through the original model before
/// binning. Leaf indices are row-major `[rows][trees]`. CSV, TSV, and LibSVM
/// all use dense-equivalent raw routing (omitted sparse features are zero).
/// Sparse parsing stores only supplied values. Dataset construction retains
/// a dense raw matrix only when `linear_tree=true` requires raw numerical values.
/// Rows times trees and rows times classes must each be at most 16 Mi entries.
/// The caller supplies matching objective, feature types, and training options;
/// `Model::refit` checks model compatibility. Category codebooks remain caller-owned.
pub fn load_refit_dataset(
    reader: impl Read,
    format: InputFormat,
    shape: InputShape,
    model: &Model,
    config: &Config,
) -> Result<(Dataset, Vec<Vec<usize>>)> {
    if shape.features != model.num_features() {
        return Err(invalid("declared feature count does not match model"));
    }
    if model.trees().is_empty() {
        return Err(invalid("refit requires a nonempty model"));
    }
    for width in [model.trees().len(), model.num_classes()] {
        if shape
            .rows
            .checked_mul(width)
            .is_none_or(|n| n > MAX_REFIT_ENTRIES)
        {
            return Err(invalid(
                "refit rows times trees/classes exceeds 16777216 entries",
            ));
        }
    }
    load_labeled_dataset(reader, format, shape, config, Some(model))
}

fn load_labeled_dataset(
    reader: impl Read,
    format: InputFormat,
    shape: InputShape,
    config: &Config,
    model: Option<&Model>,
) -> Result<(Dataset, Vec<Vec<usize>>)> {
    config.validate()?;
    let mut leaves = Vec::new();
    let mut labels = Vec::new();
    let mut values = Vec::new();
    let mut indices = Vec::new();
    let mut pointers = vec![0];
    read_rows(reader, format, shape, true, Some(config.objective), |row| {
        push(
            &mut labels,
            row.label
                .ok_or_else(|| invalid("labeled input requires labels"))?,
        )?;
        let additional = match &row.features {
            Features::Dense(row) => row.len(),
            Features::Sparse(row) => row.len(),
        };
        if values
            .len()
            .checked_add(additional)
            .is_none_or(|n| n > MAX_TRAIN_VALUES)
        {
            return Err(invalid(
                "training/refit input exceeds 8388608 supplied feature values",
            ));
        }
        reserve(&mut values, additional)?;
        match row.features {
            Features::Dense(row) => {
                if let Some(model) = model {
                    push(&mut leaves, model.predict_leaf_row(&row)?)?;
                }
                values.extend(row);
            }
            Features::Sparse(row) => {
                let start = values.len();
                reserve(&mut indices, row.len())?;
                for (index, value) in row {
                    indices.push(index);
                    values.push(value);
                }
                if let Some(model) = model {
                    push(
                        &mut leaves,
                        model.predict_leaf_sparse_row(&indices[start..], &values[start..])?,
                    )?;
                }
                push(&mut pointers, values.len())?;
            }
        }
        Ok(())
    })?;
    let dataset = match format {
        InputFormat::Csv | InputFormat::Tsv => {
            Dataset::from_dense(&values, shape.rows, shape.features, labels, config)
        }
        InputFormat::LibSvm => Dataset::from_csr(
            &pointers,
            &indices,
            &values,
            shape.rows,
            shape.features,
            labels,
            config,
        ),
    }?;
    Ok((dataset, leaves))
}

/// Write transformed predictions as headerless TSV, without input labels.
/// A writer can contain earlier rows if a later row fails validation. Use
/// `predict_to_file` when partial output must not be published.
pub fn predict(
    reader: impl Read,
    format: InputFormat,
    shape: InputShape,
    model: &Model,
    mut writer: impl Write,
) -> Result<()> {
    if shape.features != model.num_features() {
        return Err(invalid(&format!(
            "declared feature count {} does not match model feature count {}",
            shape.features,
            model.num_features()
        )));
    }
    read_rows(reader, format, shape, false, None, |row| {
        let values = match row.features {
            Features::Dense(values) => values,
            Features::Sparse(entries) => {
                // The model width and the fixed feature limit are checked first.
                let mut values = Vec::new();
                reserve(&mut values, shape.features)?;
                values.resize(shape.features, 0.0);
                for (index, value) in entries {
                    values[index] = value;
                }
                values
            }
        };
        let prediction = model.predict_dense(&values, 1, shape.features)?;
        for (column, value) in prediction[0].iter().enumerate() {
            if column != 0 {
                writer.write_all(b"\t")?;
            }
            write!(writer, "{value}")?;
        }
        writer.write_all(b"\n")?;
        Ok(())
    })?;
    writer.flush()?;
    Ok(())
}

/// Publish prediction output atomically without replacing any existing path.
/// The temporary file is in the destination directory; an atomic hard link
/// publishes it only after parsing, prediction, flush, and file synchronization.
pub fn predict_to_file(
    reader: impl Read,
    format: InputFormat,
    shape: InputShape,
    model: &Model,
    path: impl AsRef<Path>,
) -> Result<()> {
    let path = path.as_ref();
    if path.file_name().is_none() {
        return Err(invalid("output needs a file name"));
    }
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let mut temporary = None;
    for _ in 0..128 {
        let id = TEMP_ID.fetch_add(1, Ordering::Relaxed);
        let temp = parent.join(format!(
            ".lightgbm-prediction-{}-{id}.tmp",
            std::process::id()
        ));
        match OpenOptions::new().write(true).create_new(true).open(&temp) {
            Ok(file) => {
                temporary = Some((temp, file));
                break;
            }
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(e.into()),
        }
    }
    let (temp, file) =
        temporary.ok_or_else(|| invalid("cannot create prediction temporary file"))?;
    let result = (|| -> Result<()> {
        let mut writer = BufWriter::new(file);
        predict(reader, format, shape, model, &mut writer)?;
        writer.flush()?;
        writer.get_ref().sync_all()?;
        std::fs::hard_link(&temp, path)?;
        Ok(())
    })();
    let cleanup = std::fs::remove_file(&temp);
    result?;
    cleanup?;
    Ok(())
}

fn read_rows(
    reader: impl Read,
    format: InputFormat,
    shape: InputShape,
    training: bool,
    objective: Option<Objective>,
    mut visit: impl FnMut(Row) -> Result<()>,
) -> Result<()> {
    let mut bytes = Vec::new();
    reader
        .take((MAX_INPUT_BYTES + 1) as u64)
        .read_to_end(&mut bytes)?;
    if bytes.len() > MAX_INPUT_BYTES {
        return Err(invalid("input exceeds 64 MiB byte limit"));
    }
    let mut count = 0;
    match format {
        InputFormat::Csv | InputFormat::Tsv => {
            let delimiter = if format == InputFormat::Csv {
                b','
            } else {
                b'\t'
            };
            let mut reader = csv::ReaderBuilder::new()
                .has_headers(false)
                .flexible(true)
                .delimiter(delimiter)
                .from_reader(bytes.as_slice());
            let mut record = csv::StringRecord::new();
            while reader
                .read_record(&mut record)
                .map_err(|e| invalid(&format!("CSV/TSV record {}: {e}", count + 1)))?
            {
                check_row_count(count, shape)?;
                let offset = usize::from(training);
                if record.len() != shape.features + offset {
                    return Err(invalid(&format!(
                        "row {}: expected {} fields, found {}; headers and weight columns are not supported",
                        count + 1,
                        shape.features + offset,
                        record.len()
                    )));
                }
                let label = if training {
                    Some(parse_label(&record[0], objective, count + 1)?)
                } else {
                    None
                };
                let mut values = Vec::new();
                reserve(&mut values, shape.features)?;
                for field in record.iter().skip(offset) {
                    values.push(parse_feature(field, true, count + 1)?);
                }
                visit(Row {
                    label,
                    features: Features::Dense(values),
                })?;
                count += 1;
            }
        }
        InputFormat::LibSvm => {
            let text =
                std::str::from_utf8(&bytes).map_err(|_| invalid("LibSVM input must be UTF-8"))?;
            for line in text.split_terminator('\n') {
                check_row_count(count, shape)?;
                let mut fields = line.split_whitespace();
                let label = fields
                    .next()
                    .ok_or_else(|| invalid(&format!("row {}: blank LibSVM line", count + 1)))?;
                let label = parse_label(label, objective, count + 1)?;
                let mut entries = Vec::new();
                for field in fields {
                    let (index, value) = field.split_once(':').ok_or_else(|| {
                        invalid(&format!("row {}: expected index:value", count + 1))
                    })?;
                    let index: usize = index.parse().map_err(|_| {
                        invalid(&format!(
                            "row {}: invalid LibSVM index or overflow",
                            count + 1
                        ))
                    })?;
                    if index == 0 || index > shape.features {
                        return Err(invalid(&format!(
                            "row {}: LibSVM index must be in 1..={}",
                            count + 1,
                            shape.features
                        )));
                    }
                    if entries.len() >= shape.features {
                        return Err(invalid(&format!(
                            "row {}: too many LibSVM entries or duplicate indexes",
                            count + 1
                        )));
                    }
                    push(
                        &mut entries,
                        (index - 1, parse_feature(value, false, count + 1)?),
                    )?;
                }
                entries.sort_unstable_by_key(|&(index, _)| index);
                if entries.windows(2).any(|pair| pair[0].0 == pair[1].0) {
                    return Err(invalid(&format!(
                        "row {}: duplicate LibSVM index",
                        count + 1
                    )));
                }
                visit(Row {
                    label: Some(label),
                    features: Features::Sparse(entries),
                })?;
                count += 1;
            }
        }
    }
    if count != shape.rows {
        return Err(invalid(&format!(
            "row count mismatch: expected {}, found {count}",
            shape.rows
        )));
    }
    Ok(())
}

fn check_row_count(count: usize, shape: InputShape) -> Result<()> {
    if count >= shape.rows {
        return Err(invalid(&format!(
            "row count exceeds declared {}",
            shape.rows
        )));
    }
    Ok(())
}

fn parse_feature(text: &str, blank_missing: bool, row: usize) -> Result<f64> {
    let text = text.trim();
    if text.is_empty() && blank_missing {
        return Ok(f64::NAN);
    }
    let value: f64 = text
        .parse()
        .map_err(|_| invalid(&format!("row {row}: feature {text:?} is not a number")))?;
    if value.is_infinite() {
        return Err(invalid(&format!(
            "row {row}: feature must be finite or NaN"
        )));
    }
    Ok(value)
}

fn parse_label(text: &str, objective: Option<Objective>, row: usize) -> Result<f32> {
    let value: f64 = text
        .trim()
        .parse()
        .map_err(|_| invalid(&format!("row {row}: label is not a number")))?;
    let label = value as f32;
    if !value.is_finite() || !label.is_finite() {
        return Err(invalid(&format!(
            "row {row}: label must be finite and fit f32"
        )));
    }
    if objective == Some(Objective::Binary) && value != 0.0 && value != 1.0 {
        return Err(invalid(&format!("row {row}: binary label must be 0 or 1")));
    }
    Ok(label)
}

fn reserve<T>(values: &mut Vec<T>, additional: usize) -> Result<()> {
    values
        .try_reserve(additional)
        .map_err(|e| invalid(&format!("input allocation failed: {e}")))
}
fn push<T>(values: &mut Vec<T>, value: T) -> Result<()> {
    reserve(values, 1)?;
    values.push(value);
    Ok(())
}
fn invalid(message: &str) -> Error {
    Error::InvalidInput(message.into())
}
