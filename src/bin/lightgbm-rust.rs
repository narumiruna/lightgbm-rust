//! CLI over the same checked training, prediction, and refit APIs as the library.
use lightgbm_rust::io::{
    InputFormat, InputShape, load_dataset, load_refit_dataset, predict, predict_to_file,
};
use lightgbm_rust::{Config, Error, Model, Objective, Result, Trainer};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{self, BufWriter};
use std::path::PathBuf;
use std::process::ExitCode;

const HELP: &str = "lightgbm-rust: pure-Rust CPU training, prediction, and refit

USAGE:
  lightgbm-rust train --input PATH --format csv|tsv|libsvm --rows N --features N --model NEW_PATH [key=value ...]
  lightgbm-rust predict --input PATH --format csv|tsv|libsvm --rows N --features N --model PATH [--output NEW_PATH]
  lightgbm-rust refit --input PATH --format csv|tsv|libsvm --rows N --features N --model ORIGINAL_PATH --output NEW_PATH [--decay-rate 0.9] objective=NAME [key=value ...]
  lightgbm-rust --help

Options are order-independent and cannot repeat. Train and refit accept key=value
parameters. Predict rejects parameters. Config::from_pairs defines supported
names, aliases, and precedence: canonical names win; otherwise the shortest alias
wins, then lexical order. Repeated parameter names are errors. Unknown or
unsupported options are errors. CPU support is a subset, not full upstream parity.
Serial and supported parallel CPU execution use Config num_threads controls.

CSV/TSV: no header; training/refit columns are label then N features; prediction
columns are N features only. Blank features and NaN mean missing values.
Empty CSV/TSV lines are ignored; use \"\" for a one-field missing row.
LibSVM: label then one-based index:value pairs; omitted features are zero.
Unsorted indexes are accepted; duplicate indexes are errors. Prediction requires
and validates the LibSVM label but never writes it. No comments or query IDs.
All formats use dense-equivalent raw prediction and refit routing.
Labels must fit finite f32; binary training/refit labels must be 0 or 1. Infinity
is not a feature value. Labels and categories use numeric caller-owned codebooks.
String categories, headers, weight columns, and pandas adapters are unsupported.
Use the Rust Dataset API for weights, ranking query groups, positions, and initial
margins. Ranking train/refit is not available through this CLI.

Refit requires an explicit matching objective (or supported objective alias).
Also supply matching num_class for multiclass, sigmoid for a nondefault sigmoid,
reg_sqrt=true for sqrt models, linear_tree=true for linear models, and matching
categorical_feature indexes. Defaults apply to omitted parameters; model footer
training parameters are NOT recovered. Supply lambda_l1/lambda_l2, linear_lambda,
class weights, and other required training settings explicitly. Refitting starts
from zero margins; use the Rust API for other metadata. Decay must be finite in
[0,1]; default 0.9. Original raw rows determine leaf indices before new binning.
The Rust Model::refit API checks compatibility and supported model capabilities.
The original model is not changed, including on errors.

Exact row and feature counts are required; features must match the model.
Limits: 64 MiB input, 1000000 rows, 65536 features, and 8388608 supplied feature
values for training/refit. Refit rows times trees/classes are each at most 16777216.
Prediction processes one row at a time after a bounded read.
Prediction output is headerless TSV, one row per input, one column per model
output. Default prediction output is stdout, which can contain earlier rows on
a later error. File outputs are atomic and never replace existing paths or inputs.
";

#[derive(Clone, Copy, PartialEq, Eq)]
enum Operation {
    Train,
    Predict,
    Refit,
}

struct Arguments {
    operation: Operation,
    input: PathBuf,
    format: InputFormat,
    shape: InputShape,
    model: PathBuf,
    output: Option<PathBuf>,
    decay: f64,
    parameters: Vec<(String, String)>,
}

impl Arguments {
    fn parse(command: &str, args: impl Iterator<Item = String>) -> Result<Self> {
        let operation = match command {
            "train" => Operation::Train,
            "predict" => Operation::Predict,
            "refit" => Operation::Refit,
            _ => {
                return Err(Error::Unsupported(format!(
                    "command {command:?}; use --help"
                )));
            }
        };
        let mut options = BTreeMap::new();
        let mut parameters = Vec::new();
        let mut args = args;
        while let Some(arg) = args.next() {
            if arg.starts_with("--") {
                match arg.as_str() {
                    "--input" | "--format" | "--rows" | "--features" | "--model" => {}
                    "--output" if operation != Operation::Train => {}
                    "--decay-rate" if operation == Operation::Refit => {}
                    _ => return Err(Error::Unsupported(format!("option {arg:?}; use --help"))),
                }
                let value = args
                    .next()
                    .filter(|value| !value.starts_with("--"))
                    .ok_or_else(|| invalid(&format!("{arg} requires a value")))?;
                if options.insert(arg.clone(), value).is_some() {
                    return Err(invalid(&format!("duplicate option {arg}")));
                }
            } else if operation != Operation::Predict {
                let (key, value) = arg
                    .split_once('=')
                    .ok_or_else(|| invalid(&format!("expected key=value, found {arg:?}")))?;
                parameters.push((key.to_owned(), value.to_owned()));
            } else {
                return Err(invalid(&format!(
                    "unexpected argument {arg:?}; predict does not accept training parameters"
                )));
            }
        }
        let mut required = |name: &str| {
            options
                .remove(name)
                .ok_or_else(|| invalid(&format!("missing required option {name}; use --help")))
        };
        let input = PathBuf::from(required("--input")?);
        let model = PathBuf::from(required("--model")?);
        let format = required("--format")?.parse()?;
        let rows = count("--rows", &required("--rows")?)?;
        let features = count("--features", &required("--features")?)?;
        let shape = InputShape::new(rows, features)?;
        let output = options.remove("--output").map(PathBuf::from);
        if operation == Operation::Refit && output.is_none() {
            return Err(invalid("refit requires --output NEW_PATH"));
        }
        let decay = options
            .remove("--decay-rate")
            .map(|v| v.parse::<f64>())
            .transpose()
            .map_err(|_| invalid("--decay-rate requires a number"))?
            .unwrap_or(0.9);
        if !decay.is_finite() || !(0.0..=1.0).contains(&decay) {
            return Err(invalid("--decay-rate must be finite and in [0,1]"));
        }
        Ok(Self {
            operation,
            input,
            format,
            shape,
            model,
            output,
            decay,
            parameters,
        })
    }
}

fn count(name: &str, value: &str) -> Result<usize> {
    value.parse().map_err(|_| {
        invalid(&format!(
            "{name} requires an unsigned integer without overflow"
        ))
    })
}

fn cli_config(parameters: Vec<(String, String)>, refit: bool) -> Result<Config> {
    let has_objective = parameters.iter().any(|(key, value)| {
        matches!(
            key.trim(),
            "objective" | "objective_type" | "app" | "application" | "loss"
        ) && !value.trim().is_empty()
    });
    let config = Config::from_pairs(parameters)?;
    if matches!(
        config.objective,
        Objective::LambdaRank | Objective::RankXendcg
    ) {
        return Err(Error::Unsupported(
            "ranking requires query groups; use the Rust Dataset API".into(),
        ));
    }
    if refit && !has_objective {
        return Err(invalid(
            "refit requires an explicit matching objective=NAME; see --help for model metadata and training settings",
        ));
    }
    Ok(config)
}

fn run() -> Result<()> {
    let args = std::env::args_os()
        .skip(1)
        .map(|arg| {
            arg.into_string()
                .map_err(|_| invalid("arguments must be UTF-8"))
        })
        .collect::<Result<Vec<_>>>()?;
    if args.as_slice() == ["--help"] {
        print!("{HELP}");
        return Ok(());
    }
    let mut args = args.into_iter();
    let command = args
        .next()
        .ok_or_else(|| invalid("missing command; use --help"))?;
    let args = Arguments::parse(&command, args)?;
    match args.operation {
        Operation::Train => {
            let config = cli_config(args.parameters, false)?;
            let trainer = Trainer::new(config.clone())?;
            let dataset = load_dataset(File::open(&args.input)?, args.format, args.shape, &config)?;
            trainer.fit(&dataset)?.model.save(&args.model)?;
        }
        Operation::Refit => {
            let config = cli_config(args.parameters, true)?;
            let model = Model::load(&args.model)?;
            let (dataset, leaves) = load_refit_dataset(
                File::open(&args.input)?,
                args.format,
                args.shape,
                &model,
                &config,
            )?;
            let output = args
                .output
                .ok_or_else(|| invalid("refit requires --output"))?;
            model
                .refit(&dataset, &leaves, &config, args.decay)?
                .save(output)?;
        }
        Operation::Predict => {
            let model = Model::load(&args.model)?;
            let reader = File::open(&args.input)?;
            if let Some(output) = args.output {
                predict_to_file(reader, args.format, args.shape, &model, output)?;
            } else {
                predict(
                    reader,
                    args.format,
                    args.shape,
                    &model,
                    BufWriter::new(io::stdout().lock()),
                )?;
            }
        }
    }
    Ok(())
}

fn invalid(message: &str) -> Error {
    Error::InvalidInput(message.into())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("lightgbm-rust: {error}");
            ExitCode::FAILURE
        }
    }
}
