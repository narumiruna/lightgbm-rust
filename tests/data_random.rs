use lightgbm_rust::random::{Random, derive_data_seed};

#[test]
fn data_seed_is_the_first_short_draw() {
    assert_eq!(derive_data_seed(0), 38);
    assert_eq!(derive_data_seed(42), 175);
}

// Source: pinned include/LightGBM/utils/random.h. Oracle vectors are tested in data_oracle.rs.
#[test]
fn random_ranges_and_sampling_are_checked() {
    let mut random = Random::new(1);
    assert!(random.next_short(1, 1).is_err());
    assert!(random.next_int(2, 1).is_err());
    assert!(random.next_int(i32::MIN, i32::MAX).is_err());
    assert!(random.sample(usize::MAX, 1).is_err());
    assert!(random.sample(5, 0).unwrap().is_empty());
    assert!(random.sample(5, 6).unwrap().is_empty());
    assert_eq!(random.sample(5, 5).unwrap(), vec![0, 1, 2, 3, 4]);
}

#[test]
fn sampling_is_ordered_unique_bounded_and_repeatable() {
    for seed in [0, 1, 42, u32::MAX] {
        for n in [1, 2, 7, 101, 1000] {
            for k in [1, n / 2, n] {
                let a = Random::new(seed).sample(n, k).unwrap();
                let b = Random::new(seed).sample(n, k).unwrap();
                assert_eq!(a, b);
                assert_eq!(a.len(), k);
                assert!(a.windows(2).all(|w| w[0] < w[1]));
                assert!(a.iter().all(|&i| i < n));
            }
        }
    }
}

#[test]
fn floats_use_the_upstream_float32_grid() {
    let mut random = Random::new(42);
    for _ in 0..1000 {
        let value = random.next_float();
        assert!((0.0..1.0).contains(&value));
        assert_eq!((value * 32768.0).fract(), 0.0);
    }
}
