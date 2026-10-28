//! The pinned LightGBM linear congruential generator and ordered sampling.
//!
//! Ported from LightGBM include/LightGBM/utils/random.h at
//! 5d84f3f222b41409a6983d2ac90513ad9702fa9e.
//! Copyright (c) 2016-2026 Microsoft Corporation and The LightGBM developers.
//! Licensed under the MIT License; see THIRD_PARTY_NOTICES.md.

use crate::{Error, Result};
use std::collections::BTreeSet;

/// A seeded generator with the same wrapping state and draw order as LightGBM.
#[derive(Clone, Debug)]
pub struct Random {
    state: u32,
}

impl Random {
    pub fn new(seed: u32) -> Self {
        Self { state: seed }
    }

    fn advance(&mut self) -> u32 {
        self.state = self.state.wrapping_mul(214013).wrapping_add(2531011);
        self.state
    }

    fn short(&mut self) -> u32 {
        (self.advance() >> 16) & 0x7fff
    }

    /// Draw from `[lower, upper)` with the upstream 15-bit draw.
    pub fn next_short(&mut self, lower: i32, upper: i32) -> Result<i32> {
        let range = checked_range(lower, upper)?;
        Ok((self.short() % range) as i32 + lower)
    }

    /// Draw from `[lower, upper)` with the upstream 31-bit draw.
    pub fn next_int(&mut self, lower: i32, upper: i32) -> Result<i32> {
        let range = checked_range(lower, upper)?;
        Ok(((self.advance() & 0x7fff_ffff) % range) as i32 + lower)
    }

    /// Draw an `f32` from the upstream grid in `[0, 1)`.
    pub fn next_float(&mut self) -> f32 {
        self.short() as f32 / 32768.0_f32
    }

    /// Sample ordered, unique indexes. Invalid sample sizes return an empty list.
    /// Population sizes outside the upstream int32 domain return an error.
    pub fn sample(&mut self, n: usize, k: usize) -> Result<Vec<usize>> {
        if n > i32::MAX as usize {
            return Err(Error::InvalidInput(
                "sampling population exceeds int32".into(),
            ));
        }
        if k > n || k == 0 {
            return Ok(Vec::new());
        }
        let mut result = Vec::new();
        result
            .try_reserve_exact(k)
            .map_err(|e| Error::InvalidInput(format!("sample allocation failed: {e}")))?;
        if k == n {
            result.extend(0..n);
        } else if k > 1 && k as f64 > n as f64 / (k as f64).log2() {
            for i in 0..n {
                let probability = (k - result.len()) as f64 / (n - i) as f64;
                if f64::from(self.next_float()) < probability {
                    result.push(i);
                }
            }
        } else {
            let mut selected = BTreeSet::new();
            for r in n - k..n {
                let v = self.next_int(0, (r + 1) as i32)? as usize;
                if !selected.insert(v) {
                    selected.insert(r);
                }
            }
            result.extend(selected);
        }
        Ok(result)
    }
}

fn checked_range(lower: i32, upper: i32) -> Result<u32> {
    let range = upper
        .checked_sub(lower)
        .filter(|&n| n > 0)
        .ok_or_else(|| Error::InvalidInput("random range must be positive and fit int32".into()))?;
    Ok(range as u32)
}

/// Derive the dataset seed when an explicit global seed is supplied.
pub fn derive_data_seed(seed: u32) -> u32 {
    Random::new(seed).short() % 32767
}
