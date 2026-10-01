//! Re-exports the OS crate for the current `target_os`, so mainline code carries no per-OS cfg.
#![no_std]

#[cfg(target_os = "macos")]
#[allow(
    unused_imports,
    reason = "the OS crates start empty and grow with mainline needs"
)]
pub use os_darwin::*;
#[cfg(target_os = "linux")]
#[allow(
    unused_imports,
    reason = "the OS crates start empty and grow with mainline needs"
)]
pub use os_linux::*;
