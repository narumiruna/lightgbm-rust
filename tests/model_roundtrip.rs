//! Export artifacts for the isolated native interoperability check. Normal
//! Cargo tests only write and reload these models with the Rust implementation.
use lightgbm_rust::model::Model;
use serde::Deserialize;
use std::path::Path;

#[derive(Deserialize)]
struct Suite {
    cases: Vec<Case>,
}
#[derive(Deserialize)]
struct Case {
    name: String,
    model_file: String,
}

#[test]
fn export_reference_models_for_native_roundtrip() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let fixtures = root.join("tests/fixtures");
    let suite: Suite =
        serde_json::from_str(&std::fs::read_to_string(fixtures.join("prediction.json")).unwrap())
            .unwrap();
    // A new directory per test process keeps all existing artifacts intact.
    let output = root
        .join("target")
        .join(format!("model-roundtrip-{}", std::process::id()));
    std::fs::create_dir_all(output.parent().unwrap()).unwrap();
    std::fs::create_dir(&output).unwrap();
    for case in suite.cases {
        assert!(
            Path::new(&case.model_file)
                .components()
                .all(|c| matches!(c, std::path::Component::Normal(_)))
        );
        assert!(
            case.name
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '_')
        );
        let model = Model::load(fixtures.join(&case.model_file)).unwrap();
        let path = output.join(format!("{}.txt", case.name));
        model.save(&path).unwrap();
        let reloaded = Model::load(path).unwrap();
        assert_eq!(reloaded.num_features(), model.num_features());
        assert_eq!(reloaded.num_classes(), model.num_classes());
        assert_eq!(reloaded.num_iterations(), model.num_iterations());
        assert_eq!(reloaded.feature_names(), model.feature_names());
        assert_eq!(reloaded.feature_infos(), model.feature_infos());
        assert_eq!(reloaded.objective(), model.objective());
        assert_eq!(reloaded.average_output(), model.average_output());
    }
    println!("native roundtrip model directory: {}", output.display());
}
