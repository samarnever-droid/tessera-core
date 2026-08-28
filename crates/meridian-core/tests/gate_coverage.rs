//! Phase 2 gate coverage: every engine hot-path function must carry the
//! `#[bounded]` attribute, so a new hot path added without a declared step
//! bound fails this test rather than slipping through review.

const HOT_PATHS: &[&str] = &[
    "fn lookup(",
    "fn locked_find(",
    "fn locked_lookup(",
    "fn drain_retired(",
    "pub fn set_opts(",
    "pub fn get(",
    "pub fn get_ref(",
];

#[test]
fn hot_paths_carry_the_bounded_gate() {
    let src = include_str!("../src/engine.rs");
    for sig in HOT_PATHS {
        let idx = src
            .find(sig)
            .unwrap_or_else(|| panic!("{sig} not found in engine.rs"));
        let before = &src[..idx];
        let attr = before
            .rfind("#[meridian_bounded::bounded")
            .unwrap_or_else(|| panic!("{sig} has no #[bounded] attribute above it"));
        // The nearest attribute above the signature must belong to THIS fn:
        // nothing that looks like another function may appear between them.
        let between = &before[attr..];
        assert!(
            !between.contains("fn "),
            "{sig} has code between the attribute and the signature — attribute belongs elsewhere"
        );
    }
}
