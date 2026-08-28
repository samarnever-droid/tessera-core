//! 48-Bit Tagged Pointer / Metadata Packing (Zero-Cost Bit Stealing).

const PTR_MASK: u64 = 0x0000_FFFF_FFFF_FFFF;
const TAG_SHIFT: u64 = 48;

pub struct TaggedPtr;

impl TaggedPtr {
    /// Packs a 48-bit user-space pointer and a 16-bit metadata tag into a single 64-bit word.
    #[inline(always)]
    pub fn pack<T>(ptr: *const T, tag: u16) -> u64 {
        let addr = ptr as u64;
        assert_eq!(addr & !PTR_MASK, 0, "Pointer uses upper 16 bits");
        addr | ((tag as u64) << TAG_SHIFT)
    }

    /// Extracts the raw pointer from the packed word.
    #[inline(always)]
    pub fn unpack_ptr<T>(raw: u64) -> *const T {
        (raw & PTR_MASK) as *const T
    }

    /// Extracts the 16-bit metadata tag from the packed word.
    #[inline(always)]
    pub fn unpack_tag(raw: u64) -> u16 {
        (raw >> TAG_SHIFT) as u16
    }
}
