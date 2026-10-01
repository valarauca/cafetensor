//! Runtime detection, one-time tier selection, the portable fallback tier and tier enumeration.
#![cfg_attr(target_arch = "x86_64", feature(clflushopt_target_feature))]
use std::sync::LazyLock;

use generic_operations::{Engine, Kernels};
pub use generic_operations::{Format, OpError, Operations, codebook, plane_len};

mod detect;

/// Portable fallback: all core::simd defaults, compiled at the target baseline.
struct Portable;

impl Kernels for Portable {}

static PORTABLE: Engine<Portable> = Engine::new("portable");

static SELECTED: LazyLock<&'static dyn Operations> = LazyLock::new(select);

/// The selected tier. `cafetensor-lib` calls this once and keeps the handle.
pub fn operations() -> &'static dyn Operations {
    *SELECTED
}

/// Every tier this host can run, best first, portable last. For tests and benches.
pub fn available() -> Vec<&'static dyn Operations> {
    let mut tiers = native_tiers();
    tiers.push(&PORTABLE);
    tiers
}

#[cfg(target_arch = "x86_64")]
fn native_tiers() -> Vec<&'static dyn Operations> {
    let mut tiers: Vec<&'static dyn Operations> = Vec::new();
    if detect::amd64_9800x3d() {
        tiers.push(amd64_9800x3d::operations());
    }
    if detect::amd64_v4_icl() {
        tiers.push(amd64_v4_icl::operations());
    }
    if detect::amd64_v4() {
        tiers.push(amd64_v4::operations());
    }
    if detect::amd64_v3() {
        tiers.push(amd64_v3::operations());
    }
    if detect::amd64_v2() {
        tiers.push(amd64_v2::operations());
    }
    tiers
}

/// AArch64 tiers are deferred; other architectures use `portable` only.
#[cfg(not(target_arch = "x86_64"))]
fn native_tiers() -> Vec<&'static dyn Operations> {
    Vec::new()
}

/// `CAFETENSOR_TIER=<tier_name>` forces a tier for benchmarks and comparisons. An unsupported
/// tier is a hard error, never a silent fallback.
fn select() -> &'static dyn Operations {
    let tiers = available();
    match std::env::var("CAFETENSOR_TIER") {
        Ok(name) => *tiers
            .iter()
            .find(|t| t.tier_name() == name)
            .unwrap_or_else(|| panic!("tier `{name}` is not supported on this host")),
        Err(_) => tiers[0],
    }
}
