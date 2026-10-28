//! CPU AUC, average precision, and AUC-mu metrics.
//!
//! Ported from LightGBM src/metric/{binary,multiclass}_metric.hpp at
//! 5d84f3f222b41409a6983d2ac90513ad9702fa9e (MIT).
//! Copyright (c) 2016-2026 Microsoft Corporation and The LightGBM developers.
//! See THIRD_PARTY_NOTICES.md.

use super::*;

pub(super) fn binary_curve(
    labels: &[f32],
    weights: Option<&[f32]>,
    scores: &[f64],
    average_precision: bool,
) -> Result<f64> {
    let mut order = reserved(labels.len())?;
    order.extend(0..labels.len());
    order.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap());
    let total = sum_weights(weights, labels.len());
    let mut positive = 0.0;
    let mut negative = 0.0;
    let mut sum_positive = 0.0;
    let mut predicted = 0.0;
    let mut accumulation = 0.0;
    let mut threshold = scores[order[0]];
    for row in order {
        if scores[row] != threshold {
            threshold = scores[row];
            if average_precision {
                sum_positive += positive;
                predicted += positive + negative;
                let precision = sum_positive / predicted;
                accumulation += positive * precision;
            } else {
                accumulation += negative * (positive * 0.5 + sum_positive);
                sum_positive += positive;
            }
            negative = 0.0;
            positive = 0.0;
        }
        let weight = weights.map_or(1.0, |w| f64::from(w[row]));
        if labels[row] > 0.0 {
            positive += weight;
        } else {
            negative += weight;
        }
    }
    if average_precision {
        sum_positive += positive;
        predicted += positive + negative;
        accumulation += positive * (sum_positive / predicted);
    } else {
        accumulation += negative * (positive * 0.5 + sum_positive);
        sum_positive += positive;
    }
    Ok(if sum_positive <= 0.0 || sum_positive == total {
        1.0
    } else if average_precision {
        accumulation / sum_positive
    } else {
        accumulation / (sum_positive * (total - sum_positive))
    })
}

pub(super) fn auc_mu(
    config: &Config,
    labels: &[f32],
    weights: Option<&[f32]>,
    scores: &[f64],
) -> Result<f64> {
    let classes = config.num_class;
    // The reference allocates two class-squared matrices. Bound the shape even
    // though this implementation only needs one pair accumulator at a time.
    checked_len(classes, classes)?;
    let mut class_weight: Vec<f64> = zeros(classes)?;
    for (row, &label) in labels.iter().enumerate() {
        class_weight[label as usize] += weights.map_or(1.0, |w| f64::from(w[row]));
    }
    if class_weight.contains(&0.0) {
        return invalid("AUC-mu requires positive total weight in every class");
    }
    let matrix = |i: usize, j: usize| -> f64 {
        if i == j {
            0.0
        } else if config.auc_mu_weights.is_empty() {
            1.0
        } else {
            config.auc_mu_weights[i * classes + j]
        }
    };
    // Native Init sorts row indexes with std::sort, which is not stable.
    // Its equal-label permutation affects EPS clusters in the second sort.
    let mut class_order = reserved(labels.len())?;
    class_order.extend(0..labels.len());
    auc_mu_sort(&mut class_order, &|a, b| labels[a] < labels[b], true);
    let mut v = zeros(classes)?;
    let mut answer = 0.0;
    for i in 0..classes {
        for j in i + 1..classes {
            for (k, value) in v.iter_mut().enumerate() {
                *value = matrix(i, k) - matrix(j, k);
            }
            let t1 = v[i] - v[j];
            let mut distances = reserved(labels.len())?;
            // Class order is the initial order in the pinned implementation.
            for class in [i, j] {
                for &row in &class_order {
                    if labels[row] as usize != class {
                        continue;
                    }
                    let mut distance = 0.0;
                    for k in 0..classes {
                        distance += v[k] * scores[k * labels.len() + row];
                    }
                    distances.push((row, finite(t1 * distance, "AUC-mu separating distance")?));
                }
            }
            auc_mu_sort(
                &mut distances,
                &|(a, ad), (b, bd)| {
                    if (ad - bd).abs() < EPSILON {
                        labels[a] > labels[b]
                    } else {
                        ad < bd
                    }
                },
                false,
            );
            let mut num_j = 0.0;
            let mut last_distance = 0.0;
            let mut current_j = 0.0;
            let mut pair = 0.0;
            for (row, distance) in distances {
                let weight = weights.map_or(1.0, |w| f64::from(w[row]));
                if labels[row] as usize == i {
                    pair += weight
                        * if (distance - last_distance).abs() < EPSILON {
                            num_j - 0.5 * current_j
                        } else {
                            num_j
                        };
                } else {
                    num_j += weight;
                    if (distance - last_distance).abs() < EPSILON {
                        current_j += weight;
                    } else {
                        last_distance = distance;
                        current_j = weight;
                    }
                }
            }
            answer += (pair / class_weight[i]) / class_weight[j];
        }
    }
    Ok((2.0 * answer / classes as f64) / (classes - 1) as f64)
}

// AUC-mu's EPS comparator is not a strict weak order. Do not pass it to a
// Rust slice sort, or replace it with total_cmp: both change native behavior.
//
// This checked implementation follows the non-branchless std::sort path in
// Apple clang 21.0.0, libc++ 2200.27.0 (__algorithm/{sort,make_heap,
// sift_down,pop_heap,push_heap,sort_heap}.h). Ordering logic is adapted from
// the LLVM Project, Apache-2.0 WITH LLVM-exception, https://llvm.org/LICENSE.txt.
// Modified: safe indexes, bounded scans, and guarded insertion at each range
// boundary. No native code is used by the product.
//
// Contract: deterministic single-thread pinned-image ordering where native
// sentinel assumptions hold. If they fail, stop scans at the range boundary;
// C++ has no portable ordering contract for this comparator. This extension
// accepts finite distances, without assuming comparator transitivity. It does
// not claim parity with other standard libraries or parallel native sorts.
// Partition work is linear per level, depth is 2*floor(log2(n)), insertion
// ranges are <24 (incomplete insertion moves at most eight items), and the
// depth fallback is heapsort. Thus work is O(n log n), even for cyclic orders.
fn auc_mu_sort<T: Copy>(values: &mut [T], less: &impl Fn(T, T) -> bool, arithmetic: bool) {
    if values.len() > 1 {
        let depth = 2 * values.len().ilog2();
        auc_mu_introsort(values, 0, values.len(), depth, true, less, arithmetic);
    }
}

fn auc_mu_sort3<T: Copy>(a: &mut [T], x: usize, y: usize, z: usize, lt: &impl Fn(T, T) -> bool) {
    if !lt(a[y], a[x]) {
        if lt(a[z], a[y]) {
            a.swap(y, z);
            if lt(a[y], a[x]) {
                a.swap(x, y);
            }
        }
    } else if lt(a[z], a[y]) {
        a.swap(x, z);
    } else {
        a.swap(x, y);
        if lt(a[z], a[y]) {
            a.swap(y, z);
        }
    }
}

fn auc_mu_small<T: Copy>(a: &mut [T], lo: usize, hi: usize, lt: &impl Fn(T, T) -> bool) {
    if hi - lo == 2 {
        if lt(a[lo + 1], a[lo]) {
            a.swap(lo, lo + 1);
        }
    } else if hi - lo >= 3 {
        auc_mu_sort3(a, lo, lo + 1, lo + 2, lt);
        auc_mu_insert(a, lo, lo + 3, hi, usize::MAX, lt);
    }
}

// Return false after the eighth displaced item, unless it was the last item.
// Each insertion scans a finite prefix. Eight scans are O(range length).
fn auc_mu_insert<T: Copy>(
    a: &mut [T],
    lo: usize,
    first: usize,
    hi: usize,
    limit: usize,
    lt: &impl Fn(T, T) -> bool,
) -> bool {
    let mut count = 0;
    for i in first..hi {
        if lt(a[i], a[i - 1]) {
            let value = a[i];
            let mut hole = i;
            while hole > lo && lt(value, a[hole - 1]) {
                a[hole] = a[hole - 1];
                hole -= 1;
            }
            a[hole] = value;
            count += 1;
            if count == limit {
                return i + 1 == hi;
            }
        }
    }
    true
}

fn auc_mu_incomplete<T: Copy>(
    a: &mut [T],
    lo: usize,
    hi: usize,
    lt: &impl Fn(T, T) -> bool,
) -> bool {
    if hi - lo <= 5 {
        auc_mu_small(a, lo, hi, lt);
        true
    } else {
        auc_mu_sort3(a, lo, lo + 1, lo + 2, lt);
        auc_mu_insert(a, lo, lo + 3, hi, 8, lt)
    }
}

fn auc_mu_introsort<T: Copy>(
    a: &mut [T],
    mut lo: usize,
    mut hi: usize,
    mut depth: u32,
    mut leftmost: bool,
    lt: &impl Fn(T, T) -> bool,
    arithmetic: bool,
) {
    loop {
        let len = hi - lo;
        if len <= 5 {
            auc_mu_small(a, lo, hi, lt);
            return;
        }
        if len < 24 {
            auc_mu_insert(a, lo, lo + 1, hi, usize::MAX, lt);
            return;
        }
        if depth == 0 {
            auc_mu_heap(&mut a[lo..hi], lt, arithmetic);
            return;
        }
        depth -= 1;
        let mid = lo + len / 2;
        if len > 128 {
            auc_mu_sort3(a, lo, mid, hi - 1, lt);
            auc_mu_sort3(a, lo + 1, mid - 1, hi - 2, lt);
            auc_mu_sort3(a, lo + 2, mid + 1, hi - 3, lt);
            auc_mu_sort3(a, mid - 1, mid, mid + 1, lt);
            a.swap(lo, mid);
        } else {
            auc_mu_sort3(a, mid, lo, hi - 1, lt);
        }
        let equals_left = !leftmost && !lt(a[lo - 1], a[lo]);
        let pivot = a[lo];
        let mut first = lo + 1;
        let mut last = hi;
        if equals_left {
            while first < hi && !lt(pivot, a[first]) {
                first += 1;
            }
            if first < last {
                last -= 1;
                while last > lo && lt(pivot, a[last]) {
                    last -= 1;
                }
            }
        } else {
            while first < hi && lt(a[first], pivot) {
                first += 1;
            }
            // The native guarded variant stops when first == last.
            if first == lo + 1 {
                while first < last {
                    last -= 1;
                    if lt(a[last], pivot) {
                        break;
                    }
                }
            } else {
                last -= 1;
                while last > lo && !lt(a[last], pivot) {
                    last -= 1;
                }
            }
        }
        let already_partitioned = first >= last;
        while first < last {
            a.swap(first, last);
            first += 1;
            if equals_left {
                while first < hi && !lt(pivot, a[first]) {
                    first += 1;
                }
                last -= 1;
                while last > lo && lt(pivot, a[last]) {
                    last -= 1;
                }
            } else {
                while first < hi && lt(a[first], pivot) {
                    first += 1;
                }
                last -= 1;
                while last > lo && !lt(a[last], pivot) {
                    last -= 1;
                }
            }
        }
        let pos = first - 1;
        a[lo] = a[pos];
        a[pos] = pivot;
        if equals_left {
            lo = first;
            continue;
        }
        if already_partitioned {
            let left_done = auc_mu_incomplete(a, lo, pos, lt);
            let right_done = auc_mu_incomplete(a, pos + 1, hi, lt);
            if right_done {
                if left_done {
                    return;
                }
                hi = pos;
                continue;
            }
            if left_done {
                lo = pos + 1;
                continue;
            }
        }
        auc_mu_introsort(a, lo, pos, depth, leftmost, lt, arithmetic);
        leftmost = false;
        lo = pos + 1;
    }
}

fn auc_mu_sift_up<T: Copy>(a: &mut [T], mut hole: usize, lt: &impl Fn(T, T) -> bool) {
    let value = a[hole];
    while hole > 0 {
        let parent = (hole - 1) / 2;
        if !lt(a[parent], value) {
            break;
        }
        a[hole] = a[parent];
        hole = parent;
    }
    a[hole] = value;
}

fn auc_mu_heap<T: Copy>(a: &mut [T], lt: &impl Fn(T, T) -> bool, arithmetic: bool) {
    let n = a.len();
    if n < 2 {
        return;
    }
    // libc++ makes odd heaps for arithmetic values, then inserts the last.
    let size = if arithmetic && n.is_multiple_of(2) {
        n - 1
    } else {
        n
    };
    if size >= 2 {
        for root in (0..=(size - 2) / 2).rev() {
            let top = a[root];
            let mut hole = root;
            loop {
                let mut child = 2 * hole + 1;
                if child >= size {
                    break;
                }
                if child + 1 < size && lt(a[child], a[child + 1]) {
                    child += 1;
                }
                if lt(a[child], top) {
                    break;
                }
                a[hole] = a[child];
                hole = child;
            }
            a[hole] = top;
        }
    }
    if arithmetic {
        auc_mu_sift_up(a, n - 1, lt);
    }
    // libc++ pop_heap uses Floyd's downward hole, then sift_up.
    for size in (2..=n).rev() {
        let top = a[0];
        let mut hole = 0;
        loop {
            let mut child = 2 * hole + 1;
            if child >= size {
                break;
            }
            if child + 1 < size && lt(a[child], a[child + 1]) {
                child += 1;
            }
            a[hole] = a[child];
            hole = child;
        }
        if hole == size - 1 {
            a[hole] = top;
        } else {
            a[hole] = a[size - 1];
            a[size - 1] = top;
            auc_mu_sift_up(a, hole, lt);
        }
    }
}

#[cfg(test)]
mod auc_mu_sort_tests {
    use super::*;
    use std::cell::Cell;

    #[test]
    fn checked_sort_and_heap_preserve_total_orders_and_permutations() {
        for len in [0, 1, 2, 3, 4, 5, 6, 23, 24, 25, 127, 128, 129, 1025, 8192] {
            let input: Vec<_> = (0..len).map(|i| (i * 137 + 17) % 101).collect();
            let mut expected = input.clone();
            expected.sort_unstable();
            let mut actual = input.clone();
            auc_mu_sort(&mut actual, &|a, b| a < b, true);
            assert_eq!(actual, expected, "sort length {len}");
            for arithmetic in [false, true] {
                actual.clone_from(&input);
                auc_mu_heap(&mut actual, &|a, b| a < b, arithmetic);
                assert_eq!(actual, expected, "heap length {len}");
            }
        }
    }

    #[test]
    fn cyclic_comparators_have_bounded_work_and_keep_every_element() {
        let n = 32_768_usize;
        for mode in 0..4 {
            let calls = Cell::new(0_usize);
            let less = |a: usize, b: usize| {
                calls.set(calls.get() + 1);
                match mode {
                    0 => (a % 3 + 1) % 3 == b % 3,
                    1 => a != b, // No antisymmetry; tests checked scan boundaries.
                    2 => true,   // Not even irreflexive.
                    _ => false,
                }
            };
            let mut values: Vec<_> = (0..n).collect();
            auc_mu_sort(&mut values, &less, false);
            assert!(calls.get() <= 96 * n * n.ilog2() as usize);
            let mut repeated: Vec<_> = (0..n).collect();
            auc_mu_sort(&mut repeated, &less, false);
            assert_eq!(values, repeated);
            values.sort_unstable();
            assert!(values.iter().copied().eq(0..n));
        }
    }
}
