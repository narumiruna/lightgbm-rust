use lightgbm_rust::objective::{gradients, initial_score};
use lightgbm_rust::{Config, Objective};

#[test]
fn regression_preserves_reference_f32_weight_boundary() {
    let config = Config::default();
    let labels = [1.0_f32, -2.0, 0.5];
    let weights = [0.5_f32, 2.0, 0.0];
    let scores = [1.123456789, 3.2, 1.0];
    let (g, h) = gradients(&config, &labels, Some(&weights), &scores).unwrap();
    assert_eq!(
        g[0],
        ((scores[0] - f64::from(labels[0])) as f32) * weights[0]
    );
    assert_eq!(h, weights);
    assert_eq!(g[2], 0.0);
    assert_eq!(
        initial_score(&config, &labels, Some(&weights)).unwrap(),
        -1.4
    );
}

#[test]
fn binary_derivatives_match_independent_loss_differences() {
    let config = Config {
        objective: Objective::Binary,
        sigmoid: 1.5,
        ..Config::default()
    };
    let labels = [0.0_f32, 1.0, 1.0];
    let scores = [-1.2, 0.2, 1.7];
    let (g, h) = gradients(&config, &labels, None, &scores).unwrap();
    let delta = 1e-4;
    for i in 0..labels.len() {
        let loss = |s: f64| {
            (1.0 + (config.sigmoid * s).exp()).ln() - f64::from(labels[i]) * config.sigmoid * s
        };
        let first = (loss(scores[i] + delta) - loss(scores[i] - delta)) / (2.0 * delta);
        let second = (loss(scores[i] + delta) - 2.0 * loss(scores[i]) + loss(scores[i] - delta))
            / (delta * delta);
        assert!((f64::from(g[i]) - first).abs() < 1e-6);
        assert!((f64::from(h[i]) - second).abs() < 1e-6);
    }
}

#[test]
fn class_weighting_does_not_change_binary_prior() {
    let base = Config {
        objective: Objective::Binary,
        ..Config::default()
    };
    let labels = [0.0_f32, 0.0, 0.0, 1.0];
    let scaled = Config {
        scale_pos_weight: 3.0,
        ..base.clone()
    };
    assert_eq!(
        initial_score(&base, &labels, None).unwrap(),
        initial_score(&scaled, &labels, None).unwrap()
    );
    let (g, h) = gradients(&scaled, &labels, None, &[0.0; 4]).unwrap();
    assert_eq!(g, [0.5, 0.5, 0.5, -1.5]);
    assert_eq!(h, [0.25, 0.25, 0.25, 0.75]);
}

#[test]
fn objective_input_failures_are_checked() {
    let c = Config::default();
    for (labels, weights, scores) in [
        (vec![], None, vec![]),
        (vec![f32::NAN], None, vec![0.0]),
        (vec![1.0], Some(vec![0.0]), vec![0.0]),
        (vec![1.0], Some(vec![-1.0]), vec![0.0]),
        (vec![1.0], Some(vec![1.0, 2.0]), vec![0.0]),
        (vec![1.0], None, vec![f64::INFINITY]),
        (vec![1.0], None, vec![]),
    ] {
        assert!(gradients(&c, &labels, weights.as_deref(), &scores).is_err());
    }
    let binary = Config {
        objective: Objective::Binary,
        ..c
    };
    let (g, h) = gradients(&binary, &[1.0, 1.0], None, &[0.0, 0.0]).unwrap();
    assert_eq!(g, [0.0, 0.0]);
    assert_eq!(h, [0.0, 0.0]);
}
