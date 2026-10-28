#[test]
fn package_patterns_are_root_anchored() {
    // Unanchored README/LICENSE patterns recursively include ignored native
    // checkouts and Python environments. Also verify the actual package list
    // with cargo package --locked --list before packaging.
    let manifest = include_str!("../Cargo.toml");
    // Cargo expands this array to multiple lines in the verified package.
    let include = manifest
        .split_once("include = [")
        .unwrap()
        .1
        .split_once(']')
        .unwrap()
        .0;
    let patterns: Vec<_> = include.split('"').skip(1).step_by(2).collect();
    assert!(!patterns.is_empty());
    for pattern in patterns {
        assert!(
            pattern.starts_with('/'),
            "recursive package pattern: {pattern}"
        );
        assert!(!pattern.contains("references") && !pattern.contains("tools"));
    }
}
