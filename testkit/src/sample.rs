//! Seeded generation of sample tensors from a profile.

use rand::RngExt;
use rand::distr::weighted::WeightedIndex;
use rand_chacha::ChaCha8Rng;
use rand_core::{Rng, SeedableRng};

use crate::layout::{Layout, width};
use crate::profile::DtypeProfile;

/// Draws elements whose (exponent, high residual) pairs and low byte planes follow a profile.
/// A seed reproduces a sample exactly, and filling in pieces yields the same bytes as filling
/// at once, so arbitrarily large samples can be produced chunk by chunk.
pub struct Sampler {
    layout: Option<Layout>,
    width: usize,
    pairs: Vec<(u8, u8)>,
    pair_dist: Option<WeightedIndex<u64>>,
    low: Vec<Option<WeightedIndex<u64>>>,
    rng: ChaCha8Rng,
}

impl Sampler {
    /// A sampler for `profile`, seeded with `seed`.
    pub fn new(profile: &DtypeProfile, seed: u64) -> Sampler {
        let layout = Layout::of(&profile.dtype);
        let width = width(&profile.dtype).expect("known dtype");
        let pairs = profile
            .joint
            .iter()
            .map(|&[e, h, _]| (e as u8, h as u8))
            .collect();
        let pair_dist = (!profile.joint.is_empty()).then(|| {
            WeightedIndex::new(profile.joint.iter().map(|j| j[2])).expect("positive weights")
        });
        let low = profile
            .low
            .iter()
            .map(|p| {
                p.iter()
                    .any(|&c| c > 0)
                    .then(|| WeightedIndex::new(p.iter().copied()).expect("weights"))
            })
            .collect();
        Sampler {
            layout,
            width,
            pairs,
            pair_dist,
            low,
            rng: ChaCha8Rng::seed_from_u64(seed),
        }
    }

    /// Bytes per element.
    pub fn width(&self) -> usize {
        self.width
    }

    /// Fill `out` (a whole number of elements) with the next sampled elements.
    pub fn fill(&mut self, out: &mut [u8]) {
        assert_eq!(out.len() % self.width, 0, "partial element");
        let (Some(layout), Some(dist)) = (self.layout, &self.pair_dist) else {
            self.rng.fill_bytes(out);
            return;
        };
        for o in out.chunks_exact_mut(self.width) {
            let (e, h) = self.pairs[self.rng.sample(dist)];
            let mut lo = 0u32;
            for (j, d) in self.low.iter().enumerate() {
                let b = match d {
                    Some(d) => self.rng.sample(d) as u32,
                    None => self.rng.next_u32() & 0xFF,
                };
                lo |= b << (8 * j);
            }
            let v = layout.join(e, h, lo);
            o.copy_from_slice(&v.to_le_bytes()[..self.width]);
        }
    }

    /// A fresh sample of `bytes` bytes, rounded down to whole elements.
    pub fn sample(&mut self, bytes: usize) -> Vec<u8> {
        let mut out = vec![0u8; bytes / self.width * self.width];
        self.fill(&mut out);
        out
    }
}
