use super::*;

fn assert_split_bits(actual: Option<Split>, expected: Option<Split>) {
    let fields = |split: Split| {
        (
            split.feature,
            split.threshold,
            split.categories,
            split.default_left,
            [
                split.gain,
                split.left_output,
                split.right_output,
                split.left_gradient,
                split.right_gradient,
                split.left_hessian,
                split.right_hessian,
            ]
            .map(f64::to_bits),
        )
    };
    assert_eq!(actual.map(fields), expected.map(fields));
}

#[test]
fn neutral_numeric_kernel_matches_controlled_kernel_bit_for_bit() {
    for seed in 0..128 {
        let histogram: Vec<BinStats> = (0..2 + seed % 23)
            .map(|bin| BinStats {
                gradient: f64::from(((seed * 17 + bin * 13) % 41) as f32 / 7.0 - 3.0),
                hessian: f64::from(((seed * 3 + bin * 7) % 19) as f32 / 11.0),
                // Split counts must come from Hessians, not these counts.
                count: 100,
            })
            .collect();
        let gradient = histogram.iter().map(|bin| bin.gradient).sum();
        let hessian = histogram.iter().map(|bin| bin.hessian).sum();
        let config = Config {
            min_data_in_leaf: 1 + seed % 7,
            lambda_l2: [0.0, 0.1, 3.0][seed % 3],
            min_gain_to_split: if seed % 2 == 0 { 0.0 } else { 0.1 },
            ..Config::default()
        };
        for missing in [MissingType::None, MissingType::Zero, MissingType::NaN] {
            for offset in [false, true] {
                for threshold in [None, Some((seed % histogram.len()) as u32)] {
                    let control = SplitControl {
                        threshold,
                        ..SplitControl::default()
                    };
                    assert_split_bits(
                        numeric_split::<true>(
                            &histogram,
                            3,
                            missing,
                            (seed % histogram.len()) as u32,
                            offset,
                            gradient,
                            hessian,
                            83,
                            &config,
                            &control,
                        ),
                        numeric_split::<false>(
                            &histogram,
                            3,
                            missing,
                            (seed % histogram.len()) as u32,
                            offset,
                            gradient,
                            hessian,
                            83,
                            &config,
                            &control,
                        ),
                    );
                }
            }
        }
    }
}

#[test]
fn inactive_columns_do_not_change_active_histograms_in_either_mode() {
    let mut config = Config {
        min_data_in_bin: 1,
        min_data_in_leaf: 1,
        max_bin: 17,
        feature_pre_filter: false,
        ..Config::default()
    };
    let rows = 97;
    let cols = 4;
    let values: Vec<f64> = (0..rows * cols)
        .map(|i| if i % 11 < 7 { 0.0 } else { (i % 19) as f64 })
        .collect();
    let mut pointers = vec![0];
    let mut indexes = Vec::new();
    let mut nonzeros = Vec::new();
    for row in values.chunks(cols) {
        for (feature, &value) in row.iter().enumerate() {
            if value != 0.0 {
                indexes.push(feature);
                nonzeros.push(value);
            }
        }
        pointers.push(indexes.len());
    }
    let dense = Dataset::from_dense(&values, rows, cols, vec![0.0; rows], &config).unwrap();
    let sparse = Dataset::from_csr(
        &pointers,
        &indexes,
        &nonzeros,
        rows,
        cols,
        vec![0.0; rows],
        &config,
    )
    .unwrap();
    let selected: Vec<usize> = (0..rows).filter(|row| row % 3 != 0).collect();
    let gradients: Vec<f32> = (0..rows).map(|row| (row % 17) as f32 / 7.0 - 1.0).collect();
    let hessians: Vec<f32> = (0..rows)
        .map(|row| 0.25 + (row % 7) as f32 / 13.0)
        .collect();
    let active = [true, false, true, false];
    for data in [&dense, &sparse] {
        for constant in [false, true] {
            for row_wise in [false, true] {
                config.force_row_wise = row_wise;
                config.force_col_wise = !row_wise;
                let executor = crate::parallel::Executor::new(1).unwrap();
                let full = build_with_executor(
                    data, &selected, &gradients, &hessians, constant, &config, &executor, None,
                )
                .unwrap();
                let partial = build_with_executor(
                    data,
                    &selected,
                    &gradients,
                    &hessians,
                    constant,
                    &config,
                    &executor,
                    Some(&active),
                )
                .unwrap();
                for feature in 0..cols {
                    assert_eq!(partial[feature].len(), full[feature].len());
                    if active[feature] {
                        assert_eq!(partial[feature], full[feature]);
                    } else {
                        assert!(
                            partial[feature]
                                .iter()
                                .all(|bin| *bin == BinStats::default())
                        );
                    }
                }
            }
        }
    }
}
