//! Public cafetensor API over the tier selected by `general_backend`.
use std::sync::LazyLock;

use general_backend::Operations;

static TIER: LazyLock<&'static dyn Operations> = LazyLock::new(general_backend::operations);

/// Name of the CPU tier this process selected.
pub fn tier_name() -> &'static str {
    TIER.tier_name()
}
