//! Saved BinMapper::bin_info_string() outputs from the compiled pinned reference.
//! Upstream: 5d84f3f222b41409a6983d2ac90513ad9702fa9e (MIT).
//! Native generator SHA-256: b23058c409e850730d82cbbe3899149462c793e0740141e1e198e5d72a4fd45e.
//! Native executable SHA-256: ff4f578419febaa1d99de72e1adf112b24359bb71a25699ee413a0ef5ecbf58f.
//! Reference library SHA-256: 9ed1d46d2d60bac2966fd03bd81e7d192025804f6a3e3ef24c3aa483d0ec7c1e.
//! Generated with clang++ using the extended-driver compile/link record. Each
//! case passes abs(value)>kZeroThreshold or NaN to FindBin, with total=rows.len(),
//! max_bin=63, min_data_in_bin=1, min_split_data=1, pre_filter=false,
//! use_missing=true, zero_as_missing=false, and no forced endpoints.

use lightgbm_rust::{Config, Dataset};

#[test]
fn native_feature_info_strings_match_exactly_and_survive_references() {
    let adjacent = 1.0_f64.next_up().next_up();
    let cases: &[(&str, &[f64], bool, &str)] = &[
        ("integers", &[-3.0, 0.0, 1.0, 5.0], false, "[-3:5]"),
        (
            "decimal",
            &[-0.1, 0.7],
            false,
            "[-0.10000000000000001:0.69999999999999996]",
        ),
        (
            "scientific_small",
            &[1e-5, 2e-5],
            false,
            "[1.0000000000000001e-05:2.0000000000000002e-05]",
        ),
        (
            "fixed_small",
            &[1e-4, 2e-4],
            false,
            "[0.0001:0.00020000000000000001]",
        ),
        (
            "fixed_large",
            &[1e16, 2e16],
            false,
            "[10000000000000000:20000000000000000]",
        ),
        ("scientific_large", &[1e17, 2e17], false, "[1e+17:2e+17]"),
        ("all_nan", &[f64::NAN, f64::NAN], false, "[0:0]"),
        ("zeros", &[-0.0, 0.0, 0.0], false, "[0:0]"),
        (
            "adjacent",
            &[1.0, 1.0_f64.next_up(), adjacent],
            false,
            "[1.0000000000000004:1.0000000000000004]",
        ),
        (
            "negative_and_implicit_zero",
            &[-3.0, -1.0, 0.0],
            false,
            "[-3:0]",
        ),
        (
            "infinite_extrema",
            &[f64::NEG_INFINITY, f64::INFINITY, f64::NAN, 0.0],
            false,
            "[-inf:inf]",
        ),
        (
            "categories",
            &[0.0, 1.0, 1.0, 2.0, -1.0, f64::NAN],
            true,
            "-1:1:0:2",
        ),
        ("categories_all_missing", &[f64::NAN, f64::NAN], true, ""),
        ("categories_all_negative", &[-1.0, -2.0], true, ""),
    ];
    for &(name, values, categorical, expected) in cases {
        let c = Config {
            max_bin: 63,
            min_data_in_bin: 1,
            min_data_in_leaf: 1,
            feature_pre_filter: false,
            categorical_features: if categorical { vec![0] } else { vec![] },
            ..Config::default()
        };
        let d = Dataset::from_dense(values, values.len(), 1, vec![0.0; values.len()], &c).unwrap();
        assert!(!d.has_raw_values());
        assert_eq!(d.get_mapper(0).unwrap().feature_info(), expected, "{name}");
        assert_eq!(
            d.get_mapper(0).unwrap().numerical_range().is_none(),
            categorical
        );
        let v = d.validation_from_dense(&[999.0], 1, 1, vec![0.0]).unwrap();
        let s = d.subset(&[values.len() - 1]).unwrap();
        assert_eq!(v.get_mapper(0).unwrap().feature_info(), expected, "{name}");
        assert_eq!(s.get_mapper(0).unwrap().feature_info(), expected, "{name}");
    }
}

#[test]
fn forced_cuts_do_not_change_category_metadata() {
    let c = Config {
        categorical_features: vec![0],
        min_data_in_leaf: 1,
        min_data_in_bin: 1,
        ..Config::default()
    };
    let values = [0.0, 1.0, 1.0, 2.0, -1.0, f64::NAN];
    let d = Dataset::from_dense(&values, 6, 1, vec![0.0; 6], &c).unwrap();
    let forced = Config {
        forced_bins: vec![vec![-0.5, 0.5]],
        ..c
    };
    let f = Dataset::from_dense(&values, 6, 1, vec![0.0; 6], &forced).unwrap();
    assert_eq!(
        f.get_mapper(0).unwrap().feature_info(),
        d.get_mapper(0).unwrap().feature_info()
    );
    for row in 0..6 {
        assert_eq!(f.get_bin(row, 0).unwrap(), d.get_bin(row, 0).unwrap());
    }
}
