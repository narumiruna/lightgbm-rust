//! Local regression benchmarks. The fixed native budgets use the separate
//! reproducible protocol in docs/conformance.md, not Criterion estimates.
use criterion::{Criterion, criterion_group, criterion_main};
use lightgbm_rust::{Config, Dataset, Trainer};
use std::hint::black_box;
use std::time::Duration;

fn cpu(c: &mut Criterion) {
    const ROWS: usize = 2048;
    const FEATURES: usize = 8;
    let config = Config {
        num_iterations: 20,
        min_data_in_leaf: 10,
        min_data_in_bin: 1,
        feature_pre_filter: false,
        num_threads: 1,
        ..Config::default()
    };
    let values: Vec<f64> = (0..ROWS * FEATURES)
        .map(|i| ((i * 31 + i / FEATURES * 7) % 127) as f64 - 63.0)
        .collect();
    let labels: Vec<f32> = values
        .chunks_exact(FEATURES)
        .map(|r| (0.3 * r[0] - 0.2 * r[1] + r[2].abs() / 5.0) as f32)
        .collect();
    let data = Dataset::from_dense(&values, ROWS, FEATURES, labels.clone(), &config).unwrap();
    let trainer = Trainer::new(config.clone()).unwrap();
    let model = trainer.fit(&data).unwrap().model;
    let indptr: Vec<usize> = (0..=ROWS).collect();
    let indices: Vec<usize> = (0..ROWS).map(|r| r % FEATURES).collect();
    let sparse_values = vec![1.0; ROWS];
    let sparse = Dataset::from_csr(
        &indptr,
        &indices,
        &sparse_values,
        ROWS,
        FEATURES,
        labels.clone(),
        &config,
    )
    .unwrap();
    let mut group = c.benchmark_group("cpu");
    group
        .sample_size(10)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(1));
    group.bench_function("construct_dense", |b| {
        b.iter(|| {
            Dataset::from_dense(black_box(&values), ROWS, FEATURES, labels.clone(), &config)
                .unwrap()
        })
    });
    group.bench_function("construct_csr", |b| {
        b.iter(|| {
            Dataset::from_csr(
                black_box(&indptr),
                &indices,
                &sparse_values,
                ROWS,
                FEATURES,
                labels.clone(),
                &config,
            )
            .unwrap()
        })
    });
    group.bench_function("train_dense", |b| {
        b.iter(|| trainer.fit(black_box(&data)).unwrap())
    });
    group.bench_function("train_csr", |b| {
        b.iter(|| trainer.fit(black_box(&sparse)).unwrap())
    });
    group.bench_function("predict_dense", |b| {
        b.iter(|| {
            model
                .predict_dense(black_box(&values[..512 * FEATURES]), 512, FEATURES)
                .unwrap()
        })
    });
    group.finish();
}
criterion_group!(benches, cpu);
criterion_main!(benches);
