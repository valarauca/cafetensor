//! Turns committed distribution profiles into in-memory sample tensors.
//!
//! A profile records the statistical shape of the tensors of one dtype in one safetensors
//! source: the joint histogram of (exponent field, high residual) that the codec's entropy
//! model conditions on, a histogram per raw low byte plane, special-value fractions, the
//! entropy bound, and the old project's achieved ratio. No tensor data is ever committed.

pub mod layout;
pub mod profile;
pub mod sample;

use std::path::PathBuf;

pub use layout::Layout;
pub use profile::{DtypeProfile, Profile, SourceInfo};
pub use sample::Sampler;

/// The committed profile directory, `testdata/profiles/` at the workspace root.
pub fn profiles_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../testdata/profiles")
}

/// Every committed profile, sorted by file name.
pub fn load_profiles() -> Vec<Profile> {
    let mut paths: Vec<PathBuf> = std::fs::read_dir(profiles_dir())
        .expect("testdata/profiles exists")
        .map(|e| e.expect("readable directory entry").path())
        .filter(|p| p.extension().is_some_and(|e| e == "toml"))
        .collect();
    paths.sort();
    paths
        .iter()
        .map(|p| Profile::load(p).unwrap_or_else(|e| panic!("{}: {e}", p.display())))
        .collect()
}

/// Sample size in bytes: `CAFETENSOR_TEST_BYTES` if set, otherwise `default`.
pub fn test_bytes(default: usize) -> usize {
    std::env::var("CAFETENSOR_TEST_BYTES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Sample seed: `CAFETENSOR_TEST_SEED` if set, otherwise the profile's fixed seed.
pub fn test_seed(profile_seed: u64) -> u64 {
    std::env::var("CAFETENSOR_TEST_SEED")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(profile_seed)
}
