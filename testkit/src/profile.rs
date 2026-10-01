//! Distribution profiles: what is committed under `testdata/profiles/`.

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::layout::{Class, Layout};

/// One safetensors source and the profile of each dtype it contains.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Profile {
    pub source: SourceInfo,
    pub dtypes: Vec<DtypeProfile>,
}

/// Provenance of a profile.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SourceInfo {
    pub file: String,
    pub blake3: String,
    pub bytes: u64,
    pub tensors: usize,
}

/// Statistical shape of every tensor of one dtype in a source.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DtypeProfile {
    pub dtype: String,
    pub tensors: usize,
    pub elements_total: u64,
    pub elements_min: u64,
    pub elements_max: u64,
    /// Default sample seed; `CAFETENSOR_TEST_SEED` overrides it.
    pub seed: u64,
    pub zero_fraction: f64,
    pub subnormal_fraction: f64,
    pub inf_fraction: f64,
    pub nan_fraction: f64,
    /// H(exponent) of the pooled histogram, bits per element. Samples drawn from this profile
    /// compress against the pooled figures.
    pub entropy_exp_bits: f64,
    /// Pooled order-0 bound: H(exponent, high residual) plus the entropy of each low byte
    /// plane, bits per element.
    pub entropy_bits: f64,
    /// H(exponent) per tensor, weighted by element count. The codec builds one model per
    /// tensor, so this is what the old project's ratio is measured against.
    pub entropy_exp_bits_per_tensor: f64,
    /// Order-0 bound per tensor, weighted by element count.
    pub entropy_bits_per_tensor: f64,
    /// Compressed / raw size achieved by the old project on this dtype in this source.
    pub old_ratio: Option<f64>,
    /// Allowed absolute deviation of a generated sample's ratio.
    pub ratio_tolerance: f64,
    /// `[exponent, high residual, count]`, floats only.
    pub joint: Vec<[u64; 3]>,
    /// One 256-entry histogram per raw low byte plane, floats only.
    pub low: Vec<Vec<u64>>,
}

impl Profile {
    /// Parse a TOML profile.
    pub fn load(path: &Path) -> Result<Profile, String> {
        let text = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
        toml::from_str(&text).map_err(|e| e.to_string())
    }

    /// Serialize to TOML.
    pub fn to_toml(&self) -> String {
        toml::to_string(self).expect("profile serializes")
    }
}

/// Shannon entropy in bits of a histogram.
pub fn entropy(counts: impl IntoIterator<Item = u64>) -> f64 {
    let counts: Vec<u64> = counts.into_iter().filter(|&c| c > 0).collect();
    let total: u64 = counts.iter().sum();
    if total == 0 {
        return 0.0;
    }
    let t = total as f64;
    counts
        .iter()
        .map(|&c| c as f64 / t)
        .map(|p| -p * p.log2())
        .sum()
}

/// Accumulates the statistics of one dtype across tensors.
#[derive(Debug, Clone)]
pub struct Accum {
    pub layout: Option<Layout>,
    pub joint: Vec<u64>,
    pub low: Vec<[u64; 256]>,
    pub classes: [u64; 5],
    pub elements: u64,
}

impl Accum {
    /// An empty accumulator for `layout` (`None` for passthrough dtypes).
    pub fn new(layout: Option<Layout>) -> Accum {
        let planes = layout.map_or(0, |l| l.low_bytes());
        Accum {
            layout,
            joint: vec![0; 1 << 16],
            low: vec![[0; 256]; planes],
            classes: [0; 5],
            elements: 0,
        }
    }

    /// Add little-endian element bytes.
    pub fn add(&mut self, bytes: &[u8], width: usize) {
        let n = bytes.len() / width;
        self.elements += n as u64;
        let Some(l) = self.layout else { return };
        for i in 0..n {
            let v = l.load(bytes, i);
            let (e, h, _) = l.split(v);
            self.joint[(e as usize) << 8 | h as usize] += 1;
            for (j, plane) in self.low.iter_mut().enumerate() {
                plane[bytes[i * width + j] as usize] += 1;
            }
            let c = match l.class(v) {
                Class::Zero => 0,
                Class::Subnormal => 1,
                Class::Normal => 2,
                Class::Inf => 3,
                Class::Nan => 4,
            };
            self.classes[c] += 1;
        }
    }

    /// Add the counts of another accumulator of the same dtype.
    pub fn merge(&mut self, other: &Accum) {
        self.elements += other.elements;
        self.joint
            .iter_mut()
            .zip(&other.joint)
            .for_each(|(a, b)| *a += b);
        for (a, b) in self.low.iter_mut().zip(&other.low) {
            a.iter_mut().zip(b).for_each(|(x, y)| *x += y);
        }
        self.classes
            .iter_mut()
            .zip(&other.classes)
            .for_each(|(a, b)| *a += b);
    }

    /// H(exponent) and the order-0 bound, bits per element.
    pub fn entropies(&self) -> (f64, f64) {
        let mut exp = [0u64; 256];
        for (k, &c) in self.joint.iter().enumerate() {
            exp[k >> 8] += c;
        }
        let bound = entropy(self.joint.iter().copied())
            + self
                .low
                .iter()
                .map(|p| entropy(p.iter().copied()))
                .sum::<f64>();
        (entropy(exp), bound)
    }

    /// Fill the statistical fields of a profile. `per_tensor` holds the element-weighted
    /// per-tensor (H(exponent), bound).
    pub fn into_profile(
        self,
        dtype: &str,
        tensors: usize,
        min: u64,
        max: u64,
        seed: u64,
        per_tensor: (f64, f64),
    ) -> DtypeProfile {
        let n = self.elements.max(1) as f64;
        let (h_exp, h) = self.entropies();
        let floats = self.layout.is_some();
        DtypeProfile {
            dtype: dtype.to_string(),
            tensors,
            elements_total: self.elements,
            elements_min: min,
            elements_max: max,
            seed,
            zero_fraction: self.classes[0] as f64 / n,
            subnormal_fraction: self.classes[1] as f64 / n,
            inf_fraction: self.classes[3] as f64 / n,
            nan_fraction: self.classes[4] as f64 / n,
            entropy_exp_bits: if floats { h_exp } else { 0.0 },
            entropy_bits: if floats { h } else { 0.0 },
            entropy_exp_bits_per_tensor: if floats { per_tensor.0 } else { 0.0 },
            entropy_bits_per_tensor: if floats { per_tensor.1 } else { 0.0 },
            old_ratio: None,
            ratio_tolerance: 0.01,
            joint: self
                .joint
                .iter()
                .enumerate()
                .filter(|&(_, &c)| c > 0)
                .map(|(k, &c)| [(k >> 8) as u64, (k & 0xFF) as u64, c])
                .collect(),
            low: self.low.iter().map(|p| p.to_vec()).collect(),
        }
    }
}
