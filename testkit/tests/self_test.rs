//! Generated samples must match their profile's histograms and entropy, and a seed must
//! reproduce a sample exactly.

use cafetensor_testkit::profile::{Accum, Profile, entropy};
use cafetensor_testkit::{Layout, Sampler, load_profiles, test_bytes, test_seed};

/// Total variation distance between two histograms.
fn tv(a: &[u64], b: &[u64]) -> f64 {
    let (sa, sb) = (a.iter().sum::<u64>() as f64, b.iter().sum::<u64>() as f64);
    0.5 * a
        .iter()
        .zip(b)
        .map(|(&x, &y)| (x as f64 / sa - y as f64 / sb).abs())
        .sum::<f64>()
}

fn profile_joint(p: &cafetensor_testkit::DtypeProfile) -> Vec<u64> {
    let mut j = vec![0u64; 1 << 16];
    for &[e, h, c] in &p.joint {
        j[(e as usize) << 8 | h as usize] = c;
    }
    j
}

fn marginals(joint: &[u64]) -> (Vec<u64>, Vec<u64>) {
    let (mut exp, mut hi) = (vec![0u64; 256], vec![0u64; 256]);
    for (k, &c) in joint.iter().enumerate() {
        exp[k >> 8] += c;
        hi[k & 0xFF] += c;
    }
    (exp, hi)
}

#[test]
fn profiles_are_committed_and_parse() {
    let profiles = load_profiles();
    assert!(!profiles.is_empty(), "no profiles under testdata/profiles");
    for p in &profiles {
        let back: Profile = toml::from_str(&p.to_toml()).expect("profile round-trips through TOML");
        assert_eq!(&back, p, "{}", p.source.file);
        assert!(p.source.blake3.starts_with("blake3-"), "{}", p.source.file);
    }
}

#[test]
fn samples_match_their_profile() {
    for p in load_profiles() {
        for d in &p.dtypes {
            let seed = test_seed(d.seed);
            let mut s = Sampler::new(d, seed);
            let bytes = s.sample(test_bytes(16 << 20).min(d.elements_total as usize * s.width()));
            let ctx = format!("{} {} seed={seed}", p.source.file, d.dtype);
            let Some(layout) = Layout::of(&d.dtype) else {
                assert_eq!(bytes.len() % s.width(), 0, "{ctx}");
                continue;
            };
            let mut acc = Accum::new(Some(layout));
            acc.add(&bytes, layout.width);
            let want = profile_joint(d);
            let (we, wh) = marginals(&want);
            let (ge, gh) = marginals(&acc.joint);
            assert!(tv(&ge, &we) < 0.01, "{ctx}: exponent TV {}", tv(&ge, &we));
            assert!(
                tv(&gh, &wh) < 0.01,
                "{ctx}: high residual TV {}",
                tv(&gh, &wh)
            );
            for (j, plane) in d.low.iter().enumerate() {
                let t = tv(&acc.low[j], plane);
                assert!(t < 0.01, "{ctx}: low plane {j} TV {t}");
            }
            let h_joint = entropy(want.iter().copied());
            let h_sample = entropy(acc.joint.iter().copied());
            assert!(
                (h_joint - h_sample).abs() < 0.02,
                "{ctx}: H(exp,hi) {h_sample} vs {h_joint}"
            );
            let (_, bound) = acc.entropies();
            assert!(
                (bound - d.entropy_bits).abs() < 0.03,
                "{ctx}: bound {bound} vs {}",
                d.entropy_bits
            );
        }
    }
}

#[test]
fn seeds_reproduce_and_chunks_match() {
    for p in load_profiles() {
        for d in &p.dtypes {
            let seed = test_seed(d.seed);
            let w = Sampler::new(d, seed).width();
            let n = 100_003 * w;
            let a = Sampler::new(d, seed).sample(n);
            let b = Sampler::new(d, seed).sample(n);
            assert_eq!(
                a, b,
                "{} {} seed={seed} not reproducible",
                p.source.file, d.dtype
            );
            let mut chunked = vec![0u8; n];
            let mut s = Sampler::new(d, seed);
            for piece in chunked.chunks_mut(4099 * w) {
                s.fill(piece);
            }
            assert_eq!(
                a, chunked,
                "{} {} seed={seed} chunked sample differs",
                p.source.file, d.dtype
            );
            let other = Sampler::new(d, seed ^ 1).sample(n);
            assert_ne!(
                a, other,
                "{} {} different seeds collide",
                p.source.file, d.dtype
            );
        }
    }
}
