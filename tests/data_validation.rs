use lightgbm_rust::{Config, Dataset};

fn config() -> Config {
    Config {
        min_data_in_leaf: 1,
        min_data_in_bin: 1,
        ..Config::default()
    }
}

#[test]
fn rejects_invalid_dense_shapes_and_metadata() {
    let c = config();
    assert!(Dataset::from_dense(&[], 0, 1, vec![], &c).is_err());
    assert!(Dataset::from_dense(&[], usize::MAX, 2, vec![], &c).is_err());
    assert!(Dataset::from_dense(&[1.0], 1, 2, vec![0.0], &c).is_err());
    assert!(Dataset::from_dense(&[1.0], 1, 1, vec![], &c).is_err());
    assert!(Dataset::from_dense(&[1.0], 1, 1, vec![f32::NAN], &c).is_err());
    let d = Dataset::from_dense(&[1.0, 2.0], 2, 1, vec![0.0, 1.0], &c).unwrap();
    assert!(d.clone().with_weights(vec![1.0]).is_err());
    assert!(d.clone().with_weights(vec![1.0, -1.0]).is_err());
    assert!(d.clone().with_weights(vec![1.0, f32::INFINITY]).is_err());
    assert!(d.with_weights(vec![0.0, 0.0]).is_err());
}

#[test]
fn sparse_dimensions_are_bounded_before_column_allocation() {
    let config = config();
    assert!(Dataset::from_csr(&[0, 0], &[], &[], 1, 65_537, vec![0.0], &config).is_err());
    assert!(
        Dataset::from_csr(&[0, 0], &[], &[], 1, i32::MAX as usize, vec![0.0], &config).is_err()
    );
}

#[test]
fn rejects_invalid_sparse_metadata_and_duplicate_coordinates() {
    let c = config();
    for pointers in [vec![], vec![1, 1, 1], vec![0, 2, 1], vec![0, 0, 2]] {
        assert!(Dataset::from_csr(&pointers, &[0], &[1.0], 2, 1, vec![0.0; 2], &c).is_err());
    }
    assert!(Dataset::from_csr(&[0, 1], &[1], &[1.0], 1, 1, vec![0.0], &c).is_err());
    assert!(Dataset::from_csr(&[0, 2], &[0, 0], &[1.0, 2.0], 1, 1, vec![0.0], &c).is_err());
    assert!(Dataset::from_csc(&[0, 2], &[0, 0], &[1.0, 2.0], 1, 1, vec![0.0], &c).is_err());
}

#[test]
fn malformed_sparse_inputs_do_not_panic() {
    let c = config();
    for code in 0_usize..625 {
        let pointers = [code % 5, code / 5 % 5, code / 25 % 5];
        let indices = [code / 125 % 5, (code + 1) % 5];
        for len in 0..=2 {
            let outcome = std::panic::catch_unwind(|| {
                Dataset::from_csr(
                    &pointers,
                    &indices[..len],
                    &[0.0, f64::NAN][..len],
                    2,
                    2,
                    vec![0.0; 2],
                    &c,
                )
            });
            assert!(
                outcome.is_ok(),
                "pointers={pointers:?} indexes={:?}",
                &indices[..len]
            );
            let outcome = std::panic::catch_unwind(|| {
                Dataset::from_csc(
                    &pointers,
                    &indices[..len],
                    &[0.0, f64::NAN][..len],
                    2,
                    2,
                    vec![0.0; 2],
                    &c,
                )
            });
            assert!(outcome.is_ok());
        }
    }
}

#[test]
fn validation_and_subsets_share_maps_and_preserve_metadata() {
    let c = config();
    let d = Dataset::from_dense(
        &[0.0, 5.0, 1.0, 5.0, 2.0, 5.0, 3.0, 5.0],
        4,
        2,
        vec![0.0, 1.0, 2.0, 3.0],
        &c,
    )
    .unwrap()
    .with_weights(vec![1.0, 2.0, 3.0, 4.0])
    .unwrap();
    let v = d
        .validation_from_dense(&[100.0, 5.0], 1, 2, vec![1.0])
        .unwrap();
    assert!(std::ptr::eq(
        d.get_mapper(0).unwrap(),
        v.get_mapper(0).unwrap()
    ));
    assert_eq!(
        v.get_bin(0, 0).unwrap(),
        d.get_mapper(0).unwrap().bin(100.0)
    );
    assert!(d.validation_from_dense(&[0.0], 1, 1, vec![1.0]).is_err());
    let s = d.subset(&[3, 1, 3]).unwrap();
    assert!(std::ptr::eq(
        d.get_mapper(0).unwrap(),
        s.get_mapper(0).unwrap()
    ));
    assert_eq!(s.labels(), &[3.0, 1.0, 3.0]);
    assert_eq!(s.weights(), Some(&[4.0, 2.0, 4.0][..]));
    assert_eq!(s.ncols(), 2);
    assert!(s.get_mapper(1).unwrap().is_trivial());
    assert_eq!(s.get_bin(1, 0).unwrap(), d.get_bin(1, 0).unwrap());
    assert!(d.subset(&[4]).is_err());
    assert!(d.subset(&[]).is_err());
    assert_eq!(d.get_bin(0, 0).unwrap(), d.get_mapper(0).unwrap().bin(0.0));
    assert!(d.get_bin(4, 0).is_err());
    assert!(d.get_bin(0, 2).is_err());
    assert!(d.get_mapper(2).is_err());
    assert!(std::ptr::eq(
        d.get_mapper(1).unwrap(),
        s.get_mapper(1).unwrap()
    ));
}

#[test]
fn category_codes_have_checked_int32_domain() {
    let mut c = config();
    c.categorical_features = vec![0];
    assert!(Dataset::from_dense(&[2147483648.0], 1, 1, vec![0.0], &c).is_err());
    assert!(Dataset::from_dense(&[f64::INFINITY], 1, 1, vec![0.0], &c).is_err());
    let d =
        Dataset::from_dense(&[2147483646.0, -1.0, f64::NAN, 1.9], 4, 1, vec![0.0; 4], &c).unwrap();
    assert!(d.get_mapper(0).unwrap().categories().contains(&2147483646));
    assert_eq!(d.get_bin(1, 0).unwrap(), 0);
    assert_eq!(d.get_bin(2, 0).unwrap(), 0);
    assert_eq!(
        d.get_mapper(0).unwrap().bin(1.1),
        d.get_mapper(0).unwrap().bin(1.9)
    );
    c.categorical_features = vec![1];
    assert!(Dataset::from_dense(&[0.0], 1, 1, vec![0.0], &c).is_err());
}

#[test]
fn dense_csr_csc_equivalence_for_bounded_generated_inputs() {
    for seed in 0..12 {
        let rows = 13;
        let cols = 5;
        let mut dense = vec![0.0; rows * cols];
        for (i, value) in dense.iter_mut().enumerate() {
            let x = (i * 17 + seed * 13) % 23;
            *value = if x == 0 {
                f64::NAN
            } else if x < 17 {
                0.0
            } else {
                x as f64 - 20.0
            };
        }
        let mut rp = vec![0];
        let (mut ri, mut rv) = (vec![], vec![]);
        for row in 0..rows {
            for col in (0..cols).rev() {
                let v = dense[row * cols + col];
                if v != 0.0 || (row + col) % 7 == 0 {
                    ri.push(col);
                    rv.push(v);
                }
            }
            rp.push(ri.len());
        }
        let mut cp = vec![0];
        let (mut ci, mut cv) = (vec![], vec![]);
        for col in 0..cols {
            for row in (0..rows).rev() {
                let v = dense[row * cols + col];
                if v != 0.0 || (row + col) % 7 == 0 {
                    ci.push(row);
                    cv.push(v);
                }
            }
            cp.push(ci.len());
        }
        for (use_missing, zero_as_missing) in [(true, false), (true, true), (false, false)] {
            let c = Config {
                max_bin: 7,
                use_missing,
                zero_as_missing,
                ..config()
            };
            let d = Dataset::from_dense(&dense, rows, cols, vec![0.0; rows], &c).unwrap();
            let csr = Dataset::from_csr(&rp, &ri, &rv, rows, cols, vec![0.0; rows], &c).unwrap();
            let csc = Dataset::from_csc(&cp, &ci, &cv, rows, cols, vec![0.0; rows], &c).unwrap();
            for row in 0..rows {
                for col in 0..cols {
                    assert_eq!(d.get_bin(row, col).unwrap(), csr.get_bin(row, col).unwrap());
                    assert_eq!(d.get_bin(row, col).unwrap(), csc.get_bin(row, col).unwrap());
                }
            }
        }
    }
}
