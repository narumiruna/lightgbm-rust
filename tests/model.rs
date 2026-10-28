use lightgbm_rust::{Error, model::Model};

fn basic() -> String {
    "tree\nversion=v4\nnum_class=1\nnum_tree_per_iteration=1\nlabel_index=0\nmax_feature_idx=0\nobjective=regression\nfeature_names=x\nfeature_infos=[0:1]\n\nTree=0\nnum_leaves=2\nnum_cat=0\nsplit_feature=0\nthreshold=0.5\ndecision_type=2\nleft_child=-1\nright_child=-2\nleaf_value=1 2\nleaf_weight=1 1\nleaf_count=1 1\ninternal_value=1.5\ninternal_weight=2\ninternal_count=2\nsplit_gain=1\nis_linear=0\nshrinkage=1\n\nend of trees\n".to_owned()
}

#[test]
fn infinite_numeric_thresholds_preserve_text_and_clamp_only_json_inspection() {
    let model = Model::from_text(&basic().replace("threshold=0.5", "threshold=inf")).unwrap();
    assert_eq!(model.trees()[0].thresholds(), [f64::INFINITY]);
    assert!(model.to_text().unwrap().contains("threshold=inf"));
    assert_eq!(model.predict_raw_row(&[1e301]).unwrap(), [1.0]);
    let json = serde_json::to_value(&model).unwrap();
    assert_eq!(json["trees"][0]["thresholds"][0], 1e300);
    assert!(Model::from_text(&basic().replace("threshold=0.5", "threshold=NaN")).is_err());
}

#[test]
fn rejects_malformed_arrays_and_indexes() {
    let valid = basic();
    Model::from_text(&valid).unwrap();
    for (from, to) in [
        ("leaf_value=1 2", "leaf_value=1"),
        ("leaf_value=1 2", "leaf_value=1 2 3"),
        ("left_child=-1", "left_child=-3"),
        ("left_child=-1", "left_child=1"),
        ("left_child=-1", "left_child=-2147483648"),
        ("split_feature=0", "split_feature=1"),
        ("split_feature=0", "split_feature=-1"),
        ("threshold=0.5", "threshold=NaN"),
        ("leaf_value=1 2", "leaf_value=inf 2"),
        ("shrinkage=1", "shrinkage=inf"),
        ("decision_type=2", "decision_type=14"),
        ("decision_type=2", "decision_type=128"),
        ("leaf_count=1 1", "leaf_count=-1 1"),
        ("num_leaves=2", "num_leaves=18446744073709551615"),
        ("max_feature_idx=0", "max_feature_idx=2147483647"),
        ("num_class=1", "num_class=2147483647"),
        ("num_tree_per_iteration=1", "num_tree_per_iteration=0"),
        ("leaf_weight=1 1", "leaf_weight=1 -1"),
        ("num_cat=0", "num_cat=0\ncat_boundaries=0 1"),
        ("num_cat=0", "num_cat=0\ncat_threshold=1"),
        ("num_leaves=2", "num_leaves=2\nnum_leaves=2"),
    ] {
        assert!(
            Model::from_text(&valid.replace(from, to)).is_err(),
            "accepted {to}"
        );
    }
}

#[test]
fn rejects_cycles_unreachable_nodes_and_shared_ownership() {
    for (from, to) in [
        ("left_child=-1", "left_child=0"),
        ("right_child=-2", "right_child=-1"),
    ] {
        assert!(Model::from_text(&basic().replace(from, to)).is_err());
    }
    let text = basic()
        .replace("num_leaves=2", "num_leaves=3")
        .replace("split_feature=0", "split_feature=0 0")
        .replace("threshold=0.5", "threshold=0.5 0.2")
        .replace("decision_type=2", "decision_type=2 2")
        .replace("left_child=-1", "left_child=-1 1")
        .replace("right_child=-2", "right_child=-2 -3")
        .replace("leaf_value=1 2", "leaf_value=1 2 3")
        .replace("leaf_weight=1 1", "leaf_weight=1 1 1")
        .replace("leaf_count=1 1", "leaf_count=1 1 1")
        .replace("internal_value=1.5", "internal_value=1.5 2")
        .replace("internal_weight=2", "internal_weight=2 1")
        .replace("internal_count=2", "internal_count=2 1")
        .replace("split_gain=1", "split_gain=1 1");
    assert!(Model::from_text(&text).is_err()); // disconnected self-cycle
    assert!(Model::from_text(&text.replace("left_child=-1 1", "left_child=1 1")).is_err());
    assert!(
        Model::from_text(
            &text
                .replace("left_child=-1 1", "left_child=1 -1")
                .replace("right_child=-2 -3", "right_child=1 -3")
        )
        .is_err()
    );
}

#[test]
fn rejects_bad_categories_versions_and_unsupported_capabilities() {
    let cat = basic()
        .replace(
            "num_cat=0",
            "num_cat=1\ncat_boundaries=0 1\ncat_threshold=2",
        )
        .replace("decision_type=2", "decision_type=1")
        .replace("threshold=0.5", "threshold=0");
    Model::from_text(&cat).unwrap();
    for (from, to) in [
        ("threshold=0", "threshold=1"),
        ("threshold=0", "threshold=0.5"),
        ("cat_boundaries=0 1", "cat_boundaries=1 1"),
        ("cat_boundaries=0 1", "cat_boundaries=0 2"),
        ("cat_threshold=2", "cat_threshold=4294967296"),
    ] {
        assert!(Model::from_text(&cat.replace(from, to)).is_err(), "{to}");
    }
    for (from, to) in [
        ("version=v4", "version=v99"),
        ("objective=regression", "objective=unknown"),
    ] {
        assert!(matches!(
            Model::from_text(&basic().replace(from, to)),
            Err(Error::Unsupported(_))
        ));
    }
}

#[test]
fn raw_category_overflow_is_unseen_not_a_saturating_integer_alias() {
    let cat = basic()
        .replace(
            "num_cat=0",
            "num_cat=1\ncat_boundaries=0 1\ncat_threshold=2",
        )
        .replace("decision_type=2", "decision_type=1")
        .replace("threshold=0.5", "threshold=0");
    let model = Model::from_text(&cat).unwrap();
    for value in [
        f64::NAN,
        -1.0,
        i32::MAX as f64 + 1.0,
        1e100,
        f64::INFINITY,
        f64::NEG_INFINITY,
    ] {
        assert_eq!(model.predict_leaf_row(&[value]).unwrap(), vec![1]);
    }
    assert_eq!(model.predict_leaf_row(&[1.9]).unwrap(), vec![0]);
    assert_eq!(model.predict_leaf_row(&[-0.5]).unwrap(), vec![1]);
}

#[test]
fn inspection_exposes_structured_read_only_arrays() {
    let model = Model::from_text(&basic()).unwrap();
    let inspected = serde_json::to_value(&model).unwrap();
    assert_eq!(inspected["trees"][0]["left_children"][0], -1);
    assert_eq!(
        inspected["trees"][0]["leaf_values"],
        serde_json::json!([1.0, 2.0])
    );
    assert_eq!(model.trees()[0].leaf_values(), &[1.0, 2.0]);
    assert_eq!(model.feature_names(), &["x"]);
}

#[test]
fn rejects_truncation_and_forged_tree_sizes() {
    let valid = basic();
    for end in 0..valid.find("shrinkage=").unwrap() {
        assert!(
            Model::from_text(&valid[..end]).is_err(),
            "accepted prefix {end}"
        );
    }
    for sizes in ["1", "18446744073709551615", "100 200", "0"] {
        assert!(
            Model::from_text(&valid.replace(
                "feature_infos=[0:1]",
                &format!("feature_infos=[0:1]\ntree_sizes={sizes}")
            ))
            .is_err()
        );
    }
}

#[test]
fn preserves_metadata_and_computes_utf8_tree_byte_lengths() {
    let text = basic()
        .replace("feature_names=x", "feature_names=温度")
        .replace(
            "feature_infos=[0:1]",
            "feature_infos=[0:1]\nmonotone_constraints=1",
        )
        + "\nparameters:\n[seed: 42]\nend of parameters\npandas_categorical:[[\"a\",\"b\"]]\n";
    let model = Model::from_text(&text).unwrap();
    let exported = model.to_text().unwrap();
    assert!(exported.contains("feature_names=温度"));
    assert!(exported.contains("monotone_constraints=1"));
    assert!(exported.contains("[seed: 42]"));
    assert!(exported.contains("pandas_categorical:"));
    let declared: usize = exported
        .lines()
        .find_map(|l| l.strip_prefix("tree_sizes="))
        .unwrap()
        .parse()
        .unwrap();
    let start = exported.find("Tree=0\n").unwrap();
    let end = exported.find("end of trees").unwrap();
    assert_eq!(declared, exported[start..end].len());
    Model::from_text(&exported).unwrap();
}

#[test]
fn atomic_save_never_clobbers_an_existing_destination_or_source() {
    let dir = std::env::temp_dir().join(format!("lightgbm-rust-model-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let source = dir.join("source.txt");
    let output = dir.join("output.txt");
    std::fs::write(&source, basic()).unwrap();
    let _ = std::fs::remove_file(&output);
    let model = Model::load(&source).unwrap();
    assert!(model.save(&source).is_err());
    assert_eq!(std::fs::read_to_string(&source).unwrap(), basic());
    model.save(&output).unwrap();
    let before = std::fs::read(&output).unwrap();
    assert!(model.save(&output).is_err());
    assert_eq!(std::fs::read(&output).unwrap(), before);
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn concurrent_atomic_saves_publish_one_complete_destination() {
    let dir = std::env::temp_dir().join(format!("lightgbm-rust-race-test-{}", std::process::id()));
    std::fs::create_dir(&dir).unwrap();
    let path = dir.join("model.txt");
    let model = Model::from_text(&basic()).unwrap();
    let barrier = std::sync::Barrier::new(8);
    let successes = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..8)
            .map(|_| {
                scope.spawn(|| {
                    barrier.wait();
                    model.save(&path).is_ok()
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .filter(|&success| success)
            .count()
    });
    assert_eq!(successes, 1);
    assert_eq!(
        Model::load(&path).unwrap().to_text().unwrap(),
        model.to_text().unwrap()
    );
    assert_eq!(
        std::fs::read_dir(&dir).unwrap().count(),
        1,
        "temporary files were not removed"
    );
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn prediction_shapes_and_iteration_ranges_are_checked() {
    let model = Model::from_text(&basic()).unwrap();
    assert!(model.predict_row(&[]).is_err());
    assert!(model.predict_raw_row(&[0.0, 0.0]).is_err());
    assert!(model.predict_leaf_row(&[]).is_err());
    assert!(model.iteration_range(0..0).is_err());
    assert!(model.iteration_range(1..2).is_err());
    let range = model.iteration_range(0..1).unwrap();
    assert_eq!(
        (range.start(), range.end(), range.len(), range.is_empty()),
        (0, 1, 1, false)
    );
    assert_eq!(
        model.predict_row_with_iterations(&[0.0], range).unwrap(),
        model.predict_row(&[0.0]).unwrap()
    );
    assert_eq!(
        model.predict_dense(&[0.0, 1.0], 2, 1).unwrap(),
        vec![vec![1.0], vec![2.0]]
    );
    assert!(model.predict_dense(&[0.0], 2, 1).is_err());
    assert!(model.predict_raw_dense(&[0.0], 1, 2).is_err());
    assert!(
        model
            .predict_leaf_dense(&[], usize::MAX, usize::MAX)
            .is_err()
    );
    assert!(model.predict_dense(&[], 0, 0).is_err());
    assert!(model.predict_dense(&[], 0, 1).unwrap().is_empty());
    assert_eq!(
        model.predict_sparse_row(&[], &[]).unwrap(),
        model.predict_row(&[0.0]).unwrap()
    );
    assert!(model.predict_sparse_row(&[0, 0], &[1.0, 2.0]).is_err());
    assert!(model.predict_sparse_row(&[1], &[1.0]).is_err());
    assert!(model.predict_sparse_row(&[0], &[]).is_err());
}

#[test]
fn iteration_ranges_cannot_bypass_checks_on_a_different_model() {
    let text = basic();
    let block = text[text.find("Tree=0").unwrap()..text.find("end of trees").unwrap()].to_owned();
    let extended = text.replace(
        "end of trees",
        &(block.replace("Tree=0", "Tree=1") + "end of trees"),
    );
    let larger = Model::from_text(&extended).unwrap();
    let smaller = Model::from_text(&text).unwrap();
    let range = larger.iteration_range(1..2).unwrap();
    assert!(
        smaller
            .predict_raw_row_with_iterations(&[0.0], range)
            .is_err()
    );
    assert!(smaller.predict_row_with_iterations(&[0.0], range).is_err());
    assert!(
        smaller
            .predict_leaf_row_with_iterations(&[0.0], range)
            .is_err()
    );
}

#[test]
fn deeply_nested_models_use_bounded_iterative_graph_validation_and_traversal() {
    const NODES: usize = 4096;
    let mut text = String::from(
        "tree\nversion=v4\nnum_class=1\nnum_tree_per_iteration=1\nlabel_index=0\nmax_feature_idx=0\nobjective=regression\nfeature_names=x\nfeature_infos=none\n\nTree=0\n",
    );
    text.push_str(&format!("num_leaves={}\nnum_cat=0\n", NODES + 1));
    for (key, values) in [
        ("split_feature", vec!["0".to_owned(); NODES]),
        ("threshold", vec!["0".to_owned(); NODES]),
        (
            "left_child",
            (1..=NODES)
                .map(|i| {
                    if i == NODES {
                        (!(NODES as i32)).to_string()
                    } else {
                        i.to_string()
                    }
                })
                .collect(),
        ),
        (
            "right_child",
            (0..NODES).map(|i| (!(i as i32)).to_string()).collect(),
        ),
        ("leaf_value", vec!["1".to_owned(); NODES + 1]),
    ] {
        text.push_str(&format!("{key}={}\n", values.join(" ")));
    }
    text.push_str("shrinkage=1\n\nend of trees\n");
    let model = Model::from_text(&text).unwrap();
    assert_eq!(model.predict_leaf_row(&[-1.0]).unwrap(), vec![NODES]);
    assert_eq!(model.predict_raw_row(&[-1.0]).unwrap(), vec![1.0]);
    assert!(Model::from_text(&text.replace("left_child=1 ", "left_child=0 ")).is_err());
}

#[test]
fn deterministic_bounded_reader_mutations_never_panic_or_bypass_validation() {
    let original = basic().into_bytes();
    let mut seed = 0x5d84_f3f2_22b4_1409_u64;
    for _ in 0..512 {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let mut bytes = original.clone();
        let index = seed as usize % bytes.len();
        bytes[index] = [b'0', b'9', b'-', b' ', b'\n', b'=', 0][(seed >> 32) as usize % 7];
        let text = String::from_utf8(bytes).unwrap();
        if let Ok(model) = Model::from_text(&text) {
            let reloaded = Model::from_text(&model.to_text().unwrap()).unwrap();
            let leaves = reloaded
                .predict_leaf_row(&vec![0.0; reloaded.num_features()])
                .unwrap();
            for (tree, leaf) in reloaded.trees().iter().zip(leaves) {
                assert!(leaf < tree.num_leaves());
            }
        }
    }
}

#[cfg(unix)]
#[test]
fn atomic_save_rejects_symlinks_without_changing_the_target() {
    let dir =
        std::env::temp_dir().join(format!("lightgbm-rust-symlink-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let target = dir.join("target.txt");
    let link = dir.join("link.txt");
    std::fs::write(&target, b"original").unwrap();
    std::os::unix::fs::symlink(&target, &link).unwrap();
    assert!(Model::from_text(&basic()).unwrap().save(&link).is_err());
    assert_eq!(std::fs::read(&target).unwrap(), b"original");
    assert!(
        std::fs::symlink_metadata(&link)
            .unwrap()
            .file_type()
            .is_symlink()
    );
    std::fs::remove_dir_all(dir).unwrap();
}
