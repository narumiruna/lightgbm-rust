//! MT19937 and masked rejection for NumPy legacy RandomState permutations.
//! The standard recurrence uses a 624-word state and a 397-word offset.

pub(super) struct Mt19937 {
    state: [u32; 624],
    index: usize,
}
impl Mt19937 {
    pub(super) fn new(seed: u32) -> Self {
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
    fn interval(&mut self, max: u32) -> usize {
        let mut mask = max;
        mask |= mask >> 1;
        mask |= mask >> 2;
        mask |= mask >> 4;
        mask |= mask >> 8;
        mask |= mask >> 16;
        loop {
            let value = self.next() & mask;
            if value <= max {
                return value as usize;
            }
        }
    }
    pub(super) fn shuffle<T>(&mut self, values: &mut [T]) {
        for i in (1..values.len()).rev() {
            let j = self.interval(i as u32);
            values.swap(i, j);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn numpy_legacy_seed19_permutation() {
        let mut rows = (0..24).collect::<Vec<_>>();
        Mt19937::new(19).shuffle(&mut rows);
        assert_eq!(
            rows,
            [
                15, 11, 4, 20, 12, 3, 23, 9, 0, 13, 7, 16, 6, 18, 1, 5, 17, 22, 10, 19, 8, 2, 21,
                14
            ]
        );
    }
}
