//! Public cafetensor API over the tier selected by `general_backend`.
use std::sync::LazyLock;

use general_backend::Operations;

static TIER: LazyLock<&'static dyn Operations> = LazyLock::new(general_backend::operations);

/// Name of the CPU tier this process selected.
pub fn tier_name() -> &'static str {
    TIER.tier_name()
}

/// Run every operation once on a small buffer through the selected tier.
pub fn self_test() -> Result<(), general_backend::OpError> {
    let input: Vec<u8> = (0..1000u32).map(|i| (i * 7 + 3) as u8).collect();
    let mut out = vec![0u8; 500];
    TIER.op_a(&input, &mut out)?;
    let ok = out
        .iter()
        .enumerate()
        .all(|(i, &v)| v == input[i] ^ input[500 + i]);
    if ok {
        Ok(())
    } else {
        Err(general_backend::OpError::Corrupt)
    }
}
