//! Checked metadata attachment and row selection.
//!
//! Class-major selection and feature-name normalization follow LightGBM
//! src/io/metadata.cpp and include/LightGBM/dataset.h at
//! 5d84f3f222b41409a6983d2ac90513ad9702fa9e.
//! Copyright (c) 2016-2026 Microsoft Corporation and The LightGBM developers.
//! Licensed under the MIT License; see THIRD_PARTY_NOTICES.md.

use super::{Dataset, invalid, matrix_len, reserve};
use crate::{Error, Result};
use std::{collections::BTreeSet, sync::Arc};

#[derive(Clone, Debug)]
pub(super) struct Metadata {
    pub scores: Option<Vec<f64>>,
    pub groups: Option<Vec<usize>>,
    pub positions: Option<Vec<i32>>,
    pub names: Arc<[String]>,
}

impl Metadata {
    pub fn new(ncols: usize) -> Result<Self> {
        let mut names = reserve(ncols)?;
        names.extend((0..ncols).map(|f| format!("Column_{f}")));
        Ok(Self {
            scores: None,
            groups: None,
            positions: None,
            names: names.into(),
        })
    }

    pub fn validation(&self) -> Self {
        Self {
            scores: None,
            groups: None,
            positions: None,
            names: Arc::clone(&self.names),
        }
    }

    pub fn subset(&self, rows: &[usize], nrows: usize) -> Result<Self> {
        let groups = self
            .groups
            .as_ref()
            .map(|groups| -> Result<Vec<usize>> {
                if groups.contains(&0) {
                    if rows.len() == nrows && rows.iter().copied().eq(0..nrows) {
                        let mut selected = reserve(groups.len())?;
                        selected.extend_from_slice(groups);
                        return Ok(selected);
                    }
                    return Err(Error::Unsupported(
                        "row-only subsets with empty queries cannot identify query ownership; only identity selection is supported".into(),
                    ));
                }
                let mut starts = reserve(groups.len() + 1)?;
                starts.push(0);
                for &len in groups {
                    starts.push(starts.last().copied().unwrap() + len);
                }
                let mut selected = reserve(groups.len().min(rows.len()))?;
                let mut seen = BTreeSet::new();
                let mut i = 0;
                while i < rows.len() {
                    let Ok(group) = starts.binary_search(&rows[i]) else {
                        return invalid("ranking subsets must contain complete contiguous groups");
                    };
                    let len = groups[group];
                    if !seen.insert(group)
                        || rows.len() - i < len
                        || rows[i..i + len]
                            .iter()
                            .enumerate()
                            .any(|(j, &r)| r != starts[group] + j)
                    {
                        return invalid(
                            "ranking subsets cannot split, repeat, or interleave groups",
                        );
                    }
                    selected.push(len);
                    i += len;
                }
                Ok(selected)
            })
            .transpose()?;
        let scores = self
            .scores
            .as_ref()
            .map(|scores| -> Result<Vec<f64>> {
                let classes = scores.len() / nrows;
                let mut out = reserve(matrix_len(rows.len(), classes)?)?;
                for class in 0..classes {
                    out.extend(rows.iter().map(|&row| scores[class * nrows + row]));
                }
                Ok(out)
            })
            .transpose()?;
        let positions = self
            .positions
            .as_ref()
            .map(|positions| -> Result<Vec<i32>> {
                let mut out = reserve(rows.len())?;
                out.extend(rows.iter().map(|&row| positions[row]));
                Ok(out)
            })
            .transpose()?;
        Ok(Self {
            scores,
            groups,
            positions,
            names: Arc::clone(&self.names),
        })
    }
}

impl Dataset {
    /// Attach finite initial scores in class-major order: class * nrows + row.
    /// The caller supplies the score class count; the trainer checks the objective.
    pub fn with_initial_scores(mut self, scores: Vec<f64>, num_classes: usize) -> Result<Self> {
        if num_classes == 0
            || num_classes > i32::MAX as usize
            || scores.len() != matrix_len(self.nrows, num_classes)?
            || scores.iter().any(|v| !v.is_finite())
        {
            return invalid(
                "initial scores must be finite and match rows times positive class count",
            );
        }
        self.metadata.scores = Some(scores);
        Ok(self)
    }

    pub fn initial_scores(&self) -> Option<&[f64]> {
        self.metadata.scores.as_deref()
    }

    /// Attach query lengths whose checked sum equals the row count. Zero lengths
    /// are logical empty queries and retain their query indexes. Weights with empty
    /// queries are unsupported because native query-weight calculation divides by zero.
    pub fn with_groups(mut self, groups: Vec<usize>) -> Result<Self> {
        crate::objective::validate_groups(&groups, self.nrows)?;
        if groups.contains(&0) && self.weights.is_some() {
            return Err(Error::Unsupported(
                "weights with empty queries have undefined native query weights (sum / 0)".into(),
            ));
        }
        self.metadata.groups = Some(groups);
        Ok(self)
    }

    pub fn groups(&self) -> Option<&[usize]> {
        self.metadata.groups.as_deref()
    }

    /// Attach arbitrary position identifiers. Codes have no ordinal meaning.
    pub fn with_positions(mut self, positions: Vec<i32>) -> Result<Self> {
        if positions.len() != self.nrows {
            return invalid("positions must match the row count");
        }
        self.metadata.positions = Some(positions);
        Ok(self)
    }

    pub fn positions(&self) -> Option<&[i32]> {
        self.metadata.positions.as_deref()
    }

    /// Attach unique feature names. Replace spaces with underscores, as upstream.
    /// Reject JSON delimiters and control characters that cannot round-trip in a model.
    pub fn with_feature_names(mut self, mut names: Vec<String>) -> Result<Self> {
        if names.len() != self.ncols() {
            return invalid("feature names must match the column count");
        }
        let mut seen = BTreeSet::new();
        for name in &mut names {
            if name.is_empty()
                || name
                    .chars()
                    .any(|c| c.is_control() || matches!(c, '"' | ',' | ':' | '[' | ']' | '{' | '}'))
            {
                return invalid(
                    "feature names must be nonempty and cannot contain JSON delimiters or control characters",
                );
            }
            *name = name.replace(' ', "_");
            if !seen.insert(name.clone()) {
                return invalid("feature names must be unique after space replacement");
            }
        }
        self.metadata.names = names.into();
        Ok(self)
    }

    pub fn feature_names(&self) -> &[String] {
        &self.metadata.names
    }

    /// Return a retained numerical value. Requires `linear_tree=true`.
    /// Values use native raw-storage precision (`f32`, returned as `f64`).
    /// Categorical columns and invalid indexes return an error.
    pub fn raw_value(&self, row: usize, feature: usize) -> Result<f64> {
        if row >= self.nrows || feature >= self.ncols() {
            return invalid("raw indexes exceed the dataset shape");
        }
        if self.mapper(feature).is_categorical() {
            return invalid("raw access requires a numerical feature");
        }
        self.raw
            .as_ref()
            .map(|raw| f64::from(raw[row * self.ncols() + feature]))
            .ok_or_else(|| {
                Error::InvalidInput("raw values require linear_tree=true at construction".into())
            })
    }

    pub fn has_raw_values(&self) -> bool {
        self.raw.is_some()
    }
}
