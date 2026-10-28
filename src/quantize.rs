//! Serial gradient discretization from LightGBM gradient_discretizer.cpp (MIT).
//! Copyright (c) 2022-2026 Microsoft Corporation and The LightGBM developers.
use crate::{Config, Error, Result};

pub(super) struct Quantizer {
    start: Mt19937,
    gradient_random: Vec<f64>,
    hessian_random: Vec<f64>,
}
pub(super) struct Quantized {
    pub gradient: Vec<f64>,
    pub hessian: Vec<f64>,
    pub gradient_scale: f64,
    pub hessian_scale: f64,
}
impl Quantizer {
    pub fn new(seed: u32, rows: usize) -> Self {
        let mut g = Mt19937::new(seed);
        let mut h = Mt19937::new(seed.wrapping_add(1));
        Self {
            start: Mt19937::new(seed),
            gradient_random: (0..rows).map(|_| g.real()).collect(),
            hessian_random: (0..rows).map(|_| h.real()).collect(),
        }
    }
    pub fn discretize(
        &mut self,
        gradients: &[f32],
        hessians: &[f32],
        constant: bool,
        config: &Config,
    ) -> Result<Quantized> {
        let n = gradients.len();
        if n == 0
            || hessians.len() != n
            || n != self.gradient_random.len()
            || !valid_integer_range(n, config.num_grad_quant_bins)
            || gradients.iter().any(|g| !g.is_finite())
            || hessians.iter().any(|h| !h.is_finite() || *h < 0.0)
        {
            return Err(Error::InvalidInput(
                "invalid quantization dimensions or integer accumulation range".into(),
            ));
        }
        let max_g = gradients
            .iter()
            .fold(0.0_f64, |m, &v| m.max(f64::from(v).abs()));
        let max_h = hessians
            .iter()
            .fold(0.0_f64, |m, &v| m.max(f64::from(v).abs()));
        let gs = max_g / (config.num_grad_quant_bins / 2) as f64;
        let hs = if constant {
            max_h
        } else {
            max_h / config.num_grad_quant_bins as f64
        };
        let inv_g = 1.0 / gs;
        let inv_h = 1.0 / hs;
        let start = self.start.integer(n as u32 + 1) as usize;
        let mut gradient = Vec::with_capacity(n);
        let mut hessian = Vec::with_capacity(n);
        for i in 0..n {
            let pos = (i + start) % n;
            let rg = if config.stochastic_rounding {
                self.gradient_random[pos]
            } else {
                0.5
            };
            let rh = if config.stochastic_rounding {
                self.hessian_random[pos]
            } else {
                0.5
            };
            let g = f64::from(gradients[i]);
            gradient.push(if gs == 0.0 {
                0.0
            } else {
                (g * inv_g + if g >= 0.0 { rg } else { -rg }).trunc()
            });
            hessian.push(if constant {
                1.0
            } else if hs == 0.0 {
                0.0
            } else {
                (f64::from(hessians[i]) * inv_h + rh).trunc()
            });
        }
        Ok(Quantized {
            gradient,
            hessian,
            gradient_scale: gs,
            hessian_scale: hs,
        })
    }
}
fn valid_integer_range(rows: usize, bins: usize) -> bool {
    rows > 0
        && (2..=127).contains(&bins)
        && rows
            .checked_mul(bins)
            .is_some_and(|n| n <= i32::MAX as usize)
}

// std::mt19937, with the pinned libc++ uniform distributions. The reference
// manifest records the standard library because these distributions vary.
struct Mt19937 {
    state: [u32; 624],
    index: usize,
}
impl Mt19937 {
    fn new(seed: u32) -> Self {
        let mut state = [0; 624];
        state[0] = seed;
        for i in 1..624 {
            state[i] = 1812433253_u32
                .wrapping_mul(state[i - 1] ^ (state[i - 1] >> 30))
                .wrapping_add(i as u32);
        }
        Self { state, index: 624 }
    }
    fn next(&mut self) -> u32 {
        if self.index == 624 {
            for i in 0..624 {
                let y = (self.state[i] & 0x80000000) | (self.state[(i + 1) % 624] & 0x7fffffff);
                self.state[i] = self.state[(i + 397) % 624]
                    ^ (y >> 1)
                    ^ if y & 1 != 0 { 0x9908b0df } else { 0 };
            }
            self.index = 0;
        }
        let mut y = self.state[self.index];
        self.index += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c5680;
        y ^= (y << 15) & 0xefc60000;
        y ^= y >> 18;
        y
    }
    fn real(&mut self) -> f64 {
        let lo = f64::from(self.next());
        let hi = f64::from(self.next());
        (lo + hi * 4294967296.0) / 18446744073709551616.0
    }
    fn integer(&mut self, range: u32) -> u32 {
        let bits = 32 - (range - 1).leading_zeros();
        let mask = if bits == 32 {
            u32::MAX
        } else {
            (1_u32 << bits) - 1
        };
        loop {
            let value = self.next() & mask;
            if value < range {
                return value;
            }
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn quantization_bounds_zero_scales_and_integer_overflow_checks() {
        assert!(!valid_integer_range(usize::MAX, 127));
        assert!(!valid_integer_range(i32::MAX as usize, 4));
        assert!(!valid_integer_range(32, 128));
        let config = Config::default();
        let mut quantizer = Quantizer::new(42, 4);
        let q = quantizer
            .discretize(&[0.; 4], &[0.; 4], false, &config)
            .unwrap();
        assert!(q.gradient.iter().chain(&q.hessian).all(|&v| v == 0.0));
        assert_eq!((q.gradient_scale, q.hessian_scale), (0.0, 0.0));
        let q = quantizer
            .discretize(&[-3., -1., 1., 3.], &[0., 0.25, 0.5, 1.], false, &config)
            .unwrap();
        assert!(q.gradient.iter().all(|&g| g.abs() <= 2.0 && g == g.trunc()));
        assert!(
            q.hessian
                .iter()
                .all(|&h| (0.0..=4.0).contains(&h) && h == h.trunc())
        );
        assert!(
            quantizer
                .discretize(&[f32::NAN; 4], &[1.; 4], false, &config)
                .is_err()
        );
        assert!(
            quantizer
                .discretize(&[0.; 4], &[-1.; 4], false, &config)
                .is_err()
        );
    }

    #[test]
    fn four_native_quantization_cases_match_all_integer_values() {
        let f: serde_json::Value =
            serde_json::from_str(include_str!("../tests/fixtures/sampling.json")).unwrap();
        let mut count = 0;
        for case in f["cases"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|c| c["kind"] == "quantization")
        {
            count += 1;
            let g: Vec<f32> = serde_json::from_value(case["gradients"].clone()).unwrap();
            let h: Vec<f32> = serde_json::from_value(case["hessians"].clone()).unwrap();
            let config = Config {
                num_grad_quant_bins: case["params"]["num_grad_quant_bins"].as_u64().unwrap()
                    as usize,
                stochastic_rounding: case["params"]["stochastic_rounding"].as_bool().unwrap(),
                ..Config::default()
            };
            let mut q = Quantizer::new(42, g.len());
            for step in case["steps"].as_array().unwrap() {
                let result = q
                    .discretize(&g, &h, case["constant_hessian"].as_bool().unwrap(), &config)
                    .unwrap();
                assert_eq!(result.gradient_scale, step["grad_scale"].as_f64().unwrap());
                assert_eq!(result.hessian_scale, step["hess_scale"].as_f64().unwrap());
                let actual: Vec<i64> = result
                    .hessian
                    .iter()
                    .zip(&result.gradient)
                    .flat_map(|(&h, &g)| [h as i64, g as i64])
                    .collect();
                assert_eq!(
                    actual,
                    serde_json::from_value::<Vec<i64>>(step["packed_hessian_gradient"].clone())
                        .unwrap(),
                    "{} iteration {}",
                    case["name"],
                    step["iteration"]
                );
            }
        }
        assert_eq!(count, 4);
    }
}
