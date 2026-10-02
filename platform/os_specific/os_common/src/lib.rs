//! Re-exports the OS crate for the current `target_os`, so mainline code carries no per-OS cfg.

#[cfg(target_os = "macos")]
pub use os_darwin::*;
#[cfg(target_os = "linux")]
pub use os_linux::*;
