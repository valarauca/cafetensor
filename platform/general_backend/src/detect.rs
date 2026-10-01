//! The only place in the workspace that runs CPU feature detection. Each tier checks the next
//! tier down plus every feature its flags add. `lahfsahf` and `prfchw` have no detection name
//! (see DECISIONS.md).
/// Every runtime-detectable feature in scripts/tier-features/amd64_v2.txt.
#[cfg(target_arch = "x86_64")]
pub fn amd64_v2() -> bool {
    is_x86_feature_detected!("cmpxchg16b")
        && is_x86_feature_detected!("popcnt")
        && is_x86_feature_detected!("sse3")
        && is_x86_feature_detected!("sse4.1")
        && is_x86_feature_detected!("sse4.2")
        && is_x86_feature_detected!("ssse3")
}

/// Every runtime-detectable feature in scripts/tier-features/amd64_v3.txt.
#[cfg(target_arch = "x86_64")]
pub fn amd64_v3() -> bool {
    amd64_v2()
        && is_x86_feature_detected!("avx")
        && is_x86_feature_detected!("avx2")
        && is_x86_feature_detected!("bmi1")
        && is_x86_feature_detected!("bmi2")
        && is_x86_feature_detected!("f16c")
        && is_x86_feature_detected!("fma")
        && is_x86_feature_detected!("lzcnt")
        && is_x86_feature_detected!("movbe")
        && is_x86_feature_detected!("xsave")
}

/// Every runtime-detectable feature in scripts/tier-features/amd64_v4.txt.
#[cfg(target_arch = "x86_64")]
pub fn amd64_v4() -> bool {
    amd64_v3()
        && is_x86_feature_detected!("avx512bw")
        && is_x86_feature_detected!("avx512cd")
        && is_x86_feature_detected!("avx512dq")
        && is_x86_feature_detected!("avx512f")
        && is_x86_feature_detected!("avx512vl")
}

/// Every runtime-detectable feature in scripts/tier-features/amd64_v4_icl.txt.
#[cfg(target_arch = "x86_64")]
pub fn amd64_v4_icl() -> bool {
    amd64_v4()
        && is_x86_feature_detected!("avx512vbmi")
        && is_x86_feature_detected!("avx512vbmi2")
        && is_x86_feature_detected!("pclmulqdq")
        && is_x86_feature_detected!("vpclmulqdq")
}

/// Every runtime-detectable feature in scripts/tier-features/amd64_9800x3d.txt.
#[cfg(target_arch = "x86_64")]
pub fn amd64_9800x3d() -> bool {
    amd64_v4_icl()
        && is_x86_feature_detected!("adx")
        && is_x86_feature_detected!("aes")
        && is_x86_feature_detected!("avx512bf16")
        && is_x86_feature_detected!("avx512bitalg")
        && is_x86_feature_detected!("avx512ifma")
        && is_x86_feature_detected!("avx512vnni")
        && is_x86_feature_detected!("avx512vp2intersect")
        && is_x86_feature_detected!("avx512vpopcntdq")
        && is_x86_feature_detected!("avxvnni")
        && is_x86_feature_detected!("clflushopt")
        && is_x86_feature_detected!("gfni")
        && is_x86_feature_detected!("rdrand")
        && is_x86_feature_detected!("sha")
        && is_x86_feature_detected!("sse4a")
        && is_x86_feature_detected!("vaes")
        && is_x86_feature_detected!("xsavec")
        && is_x86_feature_detected!("xsaveopt")
        && is_x86_feature_detected!("xsaves")
}
