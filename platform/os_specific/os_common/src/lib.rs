//! OS facilities, the one place mainline code meets per-OS behaviour. Only Linux takes a page
//! size request for anonymous memory, so every other target gets an ordinary mapping.

use memmap2::{MmapMut, MmapOptions};

/// Map at least `len` bytes of anonymous memory and return the mapping, the offset of the
/// usable region in it, and the pages backing it. On Linux, reserved 1 GiB pages are tried
/// first when rounding up wastes at most 1/8 of `len`, then for at least 2 MiB reserved 2 MiB
/// pages and transparent huge pages, each falling back to the next, then small pages.
pub fn map_anon(len: usize) -> std::io::Result<(MmapMut, usize, &'static str)> {
    match huge(len) {
        Some(found) => Ok(found),
        None => Ok((MmapOptions::new().len(len.max(1)).map_anon()?, 0, "small")),
    }
}

#[cfg(target_os = "linux")]
fn huge(len: usize) -> Option<(MmapMut, usize, &'static str)> {
    use memmap2::Advice;
    const MIB: usize = 1 << 20;
    const GIB: usize = 1 << 30;
    let hugetlb = |page: usize, bits: u8| {
        MmapOptions::new()
            .len(len.next_multiple_of(page))
            .huge(Some(bits))
            .map_anon()
            .ok()
    };
    let thp = || {
        let n = len.next_multiple_of(2 * MIB) + 2 * MIB;
        let m = MmapOptions::new().len(n).map_anon().ok()?;
        let off = m.as_ptr().align_offset(2 * MIB).min(n);
        let _ = m.advise_range(Advice::HugePage, off, n - off);
        Some((m, off, "transparent huge"))
    };
    None.or_else(|| {
        (len.next_multiple_of(GIB) - len <= len / 8)
            .then(|| hugetlb(GIB, 30))
            .flatten()
            .map(|m| (m, 0, "1 GiB hugetlb"))
    })
    .or_else(|| {
        (len >= 2 * MIB)
            .then(|| hugetlb(2 * MIB, 21))
            .flatten()
            .map(|m| (m, 0, "2 MiB hugetlb"))
    })
    .or_else(|| (len >= 2 * MIB).then(thp).flatten())
}

#[cfg(not(target_os = "linux"))]
fn huge(_len: usize) -> Option<(MmapMut, usize, &'static str)> {
    None
}
