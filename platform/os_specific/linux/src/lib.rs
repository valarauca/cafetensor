//! Linux OS facilities. Mirrors `os_darwin` item for item.

use memmap2::{Advice, MmapMut, MmapOptions};

const MIB: usize = 1 << 20;
const GIB: usize = 1 << 30;

/// Page backing chosen for a [`HugeBuf`].
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum Backing {
    #[default]
    Empty,
    /// Reserved 1 GiB hugetlb pages.
    Huge1G,
    /// Reserved 2 MiB hugetlb pages.
    Huge2M,
    /// Transparent huge pages requested with `madvise`.
    Thp,
    /// Ordinary small pages.
    Small,
}

/// Anonymous mapping used as decode output. Large buffers are backed by huge pages so the
/// first write does not take one page fault per 4 KiB.
///
/// Policy by size: when rounding up wastes at most 1/8 of the buffer, reserved 1 GiB pages are
/// tried first (the kernel zeroes every page it hands out). At least 2 MiB tries reserved
/// 2 MiB pages and then transparent huge pages, anything else uses small pages. Each step falls
/// back to the next when the kernel refuses.
#[derive(Default)]
pub struct HugeBuf {
    map: Option<MmapMut>,
    off: usize,
    cap: usize,
    len: usize,
    backing: Backing,
}

fn hugetlb(len: usize, page: usize, bits: u8) -> Option<(MmapMut, usize, usize)> {
    let n = len.next_multiple_of(page);
    let m = MmapOptions::new().len(n).huge(Some(bits)).map_anon().ok()?;
    Some((m, 0, n))
}

fn thp(len: usize) -> Option<(MmapMut, usize, usize)> {
    let n = len.next_multiple_of(2 * MIB) + 2 * MIB;
    let m = MmapOptions::new().len(n).map_anon().ok()?;
    let off = m.as_ptr().align_offset(2 * MIB).min(n);
    let _ = m.advise_range(Advice::HugePage, off, n - off);
    Some((m, off, n - off))
}

fn small(len: usize) -> Option<(MmapMut, usize, usize)> {
    let n = len.max(1);
    let m = MmapOptions::new().len(n).map_anon().ok()?;
    Some((m, 0, n))
}

impl HugeBuf {
    /// An empty buffer.
    pub fn new() -> Self {
        Self::default()
    }

    /// Which page size backs the current mapping.
    pub fn backing(&self) -> Backing {
        self.backing
    }

    fn allocate(&mut self, len: usize) -> std::io::Result<()> {
        self.map = None;
        self.backing = Backing::Empty;
        let found = None
            .or_else(|| {
                (len.next_multiple_of(GIB) - len <= len / 8)
                    .then(|| hugetlb(len, GIB, 30))
                    .flatten()
                    .map(|m| (m, Backing::Huge1G))
            })
            .or_else(|| {
                (len >= 2 * MIB)
                    .then(|| hugetlb(len, 2 * MIB, 21))
                    .flatten()
                    .map(|m| (m, Backing::Huge2M))
            })
            .or_else(|| {
                (len >= 2 * MIB)
                    .then(|| thp(len))
                    .flatten()
                    .map(|m| (m, Backing::Thp))
            })
            .or_else(|| small(len).map(|m| (m, Backing::Small)));
        let ((map, off, cap), backing) = found.ok_or_else(std::io::Error::last_os_error)?;
        self.map = Some(map);
        self.off = off;
        self.cap = cap;
        self.backing = backing;
        Ok(())
    }

    /// Resize the usable region to `len` bytes and return it. Contents are unspecified.
    pub fn resize(&mut self, len: usize) -> std::io::Result<&mut [u8]> {
        if self.cap < len || self.map.is_none() {
            self.allocate(len)?;
        }
        self.len = len;
        let (off, len) = (self.off, self.len);
        Ok(self
            .map
            .as_mut()
            .map_or(&mut [][..], |m| &mut m[off..off + len]))
    }

    /// The usable region.
    pub fn as_slice(&self) -> &[u8] {
        self.map
            .as_ref()
            .map_or(&[][..], |m| &m[self.off..self.off + self.len])
    }
}
