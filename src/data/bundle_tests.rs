use super::*;
use serde_json::Value;

#[test]
fn native_inner_order_and_groups_match_saved_evidence() {
    let fixture: Value =
        serde_json::from_str(include_str!("../../tests/fixtures/bundling.json")).unwrap();
    for case in fixture["cases"].as_array().unwrap() {
        let pairs: Vec<_> = case["params"]
            .as_object()
            .unwrap()
            .iter()
            .map(|(k, v)| {
                (
                    k.clone(),
                    v.as_str().map_or_else(|| v.to_string(), str::to_owned),
                )
            })
            .collect();
        let cfg = Config::from_pairs(pairs).unwrap();
        let rows = case["rows"].as_array().unwrap();
        let p = rows[0].as_array().unwrap().len();
        let x: Vec<_> = rows
            .iter()
            .flat_map(|r| {
                r.as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_f64().unwrap_or(f64::NAN))
            })
            .collect();
        let data = Dataset::from_dense(&x, rows.len(), p, vec![0.; rows.len()], &cfg).unwrap();
        for f in 0..p {
            let expected = case["inner_features"][f].as_i64().unwrap();
            assert_eq!(
                data.inner_feature_index(f),
                if expected < 0 {
                    usize::MAX
                } else {
                    expected as usize
                },
                "{} feature {f}",
                case["name"]
            );
        }
        let groups: Vec<Vec<usize>> = case["groups"]
            .as_array()
            .unwrap()
            .iter()
            .map(|g| serde_json::from_value(g["features"].clone()).unwrap())
            .collect();
        assert_eq!(data.bundle_layout.groups, groups, "{}", case["name"]);
        let subset = data.subset(&[4, 2, 4, 0]).unwrap();
        assert!(Arc::ptr_eq(&data.bundle_layout, &subset.bundle_layout));
        let validation = data
            .validation_from_dense(&x, rows.len(), p, vec![0.; rows.len()])
            .unwrap();
        assert!(Arc::ptr_eq(&data.bundle_layout, &validation.bundle_layout));
    }
}

#[test]
fn packed_histograms_match_direct_and_baseline_storage() {
    let n = 240;
    let p = 12;
    let x: Vec<_> = (0..n)
        .flat_map(|i| (0..p).map(move |j| f64::from(i % p == j)))
        .collect();
    let cfg = Config {
        min_data_in_bin: 1,
        min_data_in_leaf: 1,
        ..Config::default()
    };
    let on = Dataset::from_dense(&x, n, p, vec![0.; n], &cfg).unwrap();
    let off = Dataset::from_dense(
        &x,
        n,
        p,
        vec![0.; n],
        &Config {
            enable_bundle: false,
            ..cfg
        },
    )
    .unwrap();
    let gradients: Vec<f32> = (0..n).map(|i| (i as f32 - 71.) / 13.).collect();
    let hessians = vec![1f32; n];
    let rows: Vec<_> = (0..n).filter(|i| i % 3 != 0).collect();
    let a = crate::histogram::build(&on, &rows, &gradients, &hessians, true).unwrap();
    let b = crate::histogram::build(&off, &rows, &gradients, &hessians, true).unwrap();
    assert_eq!(a, b);
    for (f, bins) in a.iter().enumerate() {
        for (b, stat) in bins.iter().enumerate() {
            let mut expected = crate::histogram::BinStats::default();
            for &r in &rows {
                if on.bin(r, f) == b as u32 {
                    expected.gradient += f64::from(gradients[r]);
                    expected.hessian += 1.;
                    expected.count += 1;
                }
            }
            assert_eq!(*stat, expected);
        }
    }
}

#[test]
fn packed_offsets_decode_each_storage_width_and_nonzero_defaults() {
    for num_bins in [17, 1024, 100_000] {
        let offset = num_bins - 7;
        let codes = [0, offset, offset + 1, offset + 4, 1];
        let packed = Arc::new(bundle::Packed {
            values: CompactBins::from_iter(codes.into_iter(), codes.len(), num_bins).unwrap(),
            num_bins,
        });
        let feature = CompactBins::Bundle {
            packed,
            offset,
            num_bins: 5,
            default: 3,
        };
        for (r, expected) in [3, 0, 1, 4, 3].into_iter().enumerate() {
            assert_eq!(feature.get(r), expected);
        }
    }
}

#[test]
fn config_fingerprint_includes_enable_bundle() {
    let cfg = Config {
        min_data_in_leaf: 1,
        min_data_in_bin: 1,
        ..Config::default()
    };
    let d = Dataset::from_dense(&[0., 1., 2., 3.], 4, 1, vec![0.; 4], &cfg).unwrap();
    assert!(
        d.check_binning_config(&Config {
            enable_bundle: false,
            ..cfg
        })
        .is_err()
    );
}
