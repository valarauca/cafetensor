//! Darwin OS facilities. Mirrors `os_linux` item for item.

use memmap2::{MmapMut, MmapOptions};

/// Page backing chosen for a [`HugeBuf`].
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum Backing {
    #[default]
    Empty,
    /// Reserved 1 GiB hugetlb pages. Never chosen on Darwin.
    Huge1G,
    /// Reserved 2 MiB hugetlb pages. Never chosen on Darwin.
    Huge2M,
    /// Transparent huge pages. Never chosen on Darwin.
    Thp,
    /// Ordinary small pages.
    Small,
}

/// Anonymous mapping used as decode output. Darwin has no huge page requests for anonymous
/// memory, so every mapping uses small pages.
#[derive(Default)]
pub struct HugeBuf {
    map: Option<MmapMut>,
    cap: usize,
    len: usize,
    backing: Backing,
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

    /// Resize the usable region to `len` bytes and return it. Contents are unspecified.
    pub fn resize(&mut self, len: usize) -> std::io::Result<&mut [u8]> {
        if self.cap < len || self.map.is_none() {
            self.map = None;
            let n = len.max(1);
            self.map = Some(MmapOptions::new().len(n).map_anon()?);
            self.cap = n;
            self.backing = Backing::Small;
        }
        self.len = len;
        Ok(self.map.as_mut().map_or(&mut [][..], |m| &mut m[..len]))
    }

    /// The usable region.
    pub fn as_slice(&self) -> &[u8] {
        self.map.as_ref().map_or(&[][..], |m| &m[..self.len])
    }
}
