//! Large output buffers on the pages the OS crate can provide.

use memmap2::MmapMut;

/// Anonymous mapping used as decode output and read buffer. On Linux large buffers are backed
/// by huge pages so the first write does not take one page fault per 4 KiB; elsewhere it is an
/// ordinary mapping. It is only remapped when it has to grow.
#[derive(Default)]
pub struct HugeBuf {
    map: Option<MmapMut>,
    off: usize,
    len: usize,
    backing: &'static str,
}

impl HugeBuf {
    /// An empty buffer.
    pub fn new() -> Self {
        Self::default()
    }

    /// The pages backing the current mapping, empty before the first [`HugeBuf::resize`].
    pub fn backing(&self) -> &'static str {
        self.backing
    }

    /// Resize the usable region to `len` bytes and return it. Contents are unspecified.
    pub fn resize(&mut self, len: usize) -> std::io::Result<&mut [u8]> {
        let fits = self.map.as_ref().is_some_and(|m| m.len() - self.off >= len);
        if !fits {
            self.map = None;
            let (map, off, backing) = os_common::map_anon(len)?;
            self.map = Some(map);
            self.off = off;
            self.backing = backing;
        }
        self.len = len;
        let off = self.off;
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
