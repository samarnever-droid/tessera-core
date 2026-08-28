//! meridian-bounded — the `#[bounded(n)]` compile gate (spec §10: "The step
//! bound becomes a type").
//!
//! Verified, on stable Rust, for any function that opts in:
//! - **structure**: no bare `loop`, no `while`/`while let` (the honest form
//!   is `for _ in 0..len_snapshot` with early `break` — that snapshot loop
//!   IS the `@decreases` measure, materialized), and `for` must iterate a
//!   range (`..`/`..=`);
//! - **numeric bound**: the deepest loop nest whose trip counts are literals
//!   or UPPER_CASE constants gets a compile-time assertion
//!   `const _: () = assert!(A * B * ... <= n);` — the constants are in scope
//!   inside the function, so the check uses the real values, not guesses.
//!   Pure-literal nests are checked directly by the macro with a clean
//!   error message.
//!
//! Not yet verified (deferred, needs the nightly rustc-driver MIR pass):
//! trip counts of runtime-bounded snapshot loops (e.g. `0..len`), and
//! sequential-sibling nests — the bound covers the deepest single nest.
//!
//! Functions without the attribute are unaffected.

use proc_macro::{Delimiter, TokenStream, TokenTree};

#[derive(Default)]
struct ScanState {
    errors: Vec<String>,
    /// One entry per `for` loop: enclosing trip-count factors × own.
    chains: Vec<Vec<String>>,
}

impl ScanState {
    /// Deepest nest made entirely of literals or UPPER_CASE (const) factors.
    fn deepest_provable(&self) -> Option<Vec<String>> {
        self.chains
            .iter()
            .filter(|c| {
                c.iter().all(|f| {
                    !f.is_empty()
                        && f.chars().all(|ch| ch.is_ascii_uppercase() || ch.is_ascii_digit() || ch == '_')
                })
            })
            .max_by_key(|c| c.len())
            .cloned()
    }
}

#[proc_macro_attribute]
pub fn bounded(attr: TokenStream, item: TokenStream) -> TokenStream {
    let Some(n) = parse_bound(&attr) else {
        return prepend_body(
            item,
            "::core::compile_error!(\"#[bounded] requires a numeric bound, e.g. #[bounded(64)]\");",
        );
    };
    let tokens: Vec<TokenTree> = item.clone().into_iter().collect();
    let mut st = ScanState::default();
    scan(&tokens, &[], &mut st);

    let mut pre = String::new();
    if let Some(factors) = st.deepest_provable() {
        let all_lit = factors.iter().all(|f| f.parse::<u128>().is_ok());
        if all_lit {
            let prod = factors
                .iter()
                .map(|f| f.parse::<u128>().unwrap())
                .try_fold(1u128, |a, b| a.checked_mul(b));
            if let Some(p) = prod {
                if p > n {
                    st.errors.push(format!(
                        "deepest loop nest {} = {p} exceeds the declared bound {n}",
                        factors.join(" * ")
                    ));
                }
            }
        } else {
            pre.push_str(&format!(
                "const _: () = assert!(({}) <= {n});",
                factors.join(" * ")
            ));
        }
    }
    if !st.errors.is_empty() {
        pre.push_str(&format!(
            "::core::compile_error!(\"{}\");",
            st.errors.join("; ").replace('\\', "\\\\").replace('"', "\\\"")
        ));
    }
    if pre.is_empty() {
        item
    } else {
        prepend_body(item, &pre)
    }
}

/// Insert statements at the top of the function body (statement position):
/// an item-position emission would be an associated item inside an impl.
fn prepend_body(item: TokenStream, pre: &str) -> TokenStream {
    let pre_ts: TokenStream = pre.parse().expect("generated code must be valid");
    let mut out = TokenStream::new();
    let mut inserted = false;
    for t in item {
        if !inserted {
            if let TokenTree::Group(g) = &t {
                if g.delimiter() == Delimiter::Brace {
                    let mut body = pre_ts.clone();
                    body.extend(g.stream());
                    out.extend(std::iter::once(TokenTree::Group(
                        proc_macro::Group::new(Delimiter::Brace, body),
                    )));
                    inserted = true;
                    continue;
                }
            }
        }
        out.extend(std::iter::once(t));
    }
    out
}

fn parse_bound(attr: &TokenStream) -> Option<u128> {
    for t in attr.clone() {
        if let TokenTree::Literal(l) = t {
            if let Ok(n) = l.to_string().replace('_', "").parse() {
                return Some(n);
            }
        }
    }
    None
}

fn scan(tokens: &[TokenTree], factors: &[String], st: &mut ScanState) {
    let mut i = 0;
    while i < tokens.len() {
        let t = &tokens[i];
        if let TokenTree::Group(g) = t {
            let inner: Vec<TokenTree> = g.stream().into_iter().collect();
            scan(&inner, factors, st);
            i += 1;
            continue;
        }
        let s = t.to_string();
        if s == "loop" {
            st.errors.push("unbounded `loop` in a #[bounded] function".into());
        } else if s == "while" {
            st.errors.push(
                "`while` in a #[bounded] function: use `for _ in 0..len_snapshot` with early `break`"
                    .into(),
            );
        } else if s == "for" {
            let mut j = i + 1;
            let mut header: Vec<TokenTree> = Vec::new();
            let mut body: Option<Vec<TokenTree>> = None;
            while j < tokens.len() {
                if let TokenTree::Group(g) = &tokens[j] {
                    if g.delimiter() == Delimiter::Brace {
                        body = Some(g.stream().into_iter().collect());
                        break;
                    }
                }
                header.push(tokens[j].clone());
                j += 1;
            }
            if !header_has_range(&header) {
                st.errors.push("`for` over a non-range iterator in a #[bounded] function".into());
            }
            let mut child = factors.to_vec();
            if let Some(f) = range_end_expr(&header) {
                child.push(f);
            }
            if !child.is_empty() {
                st.chains.push(child.clone());
            }
            if let Some(body) = body {
                scan(&body, &child, st);
            }
            i = j + 1;
            continue;
        }
        i += 1;
    }
}

/// The loop header runs from `for` to its body; a parenthesised iterable is
/// searched one level deep.
fn header_has_range(header: &[TokenTree]) -> bool {
    let mut dots = false;
    let mut prev_dot = false;
    for t in header {
        match t {
            TokenTree::Group(g) if g.delimiter() == Delimiter::Parenthesis => {
                let inner: Vec<TokenTree> = g.stream().into_iter().collect();
                let mut p = false;
                for it in inner {
                    if is_punct_dot(&it) {
                        if p {
                            dots = true;
                        }
                        p = true;
                    } else if it.to_string() != "=" {
                        p = false;
                    }
                }
                prev_dot = false;
            }
            other => {
                if is_punct_dot(other) {
                    if prev_dot {
                        dots = true;
                    }
                    prev_dot = true;
                } else {
                    prev_dot = false;
                }
            }
        }
    }
    dots
}

/// Trip-count factor of a range `for`: the token after `..` / `..=`, when it
/// is a literal or a simple identifier.
fn range_end_expr(header: &[TokenTree]) -> Option<String> {
    let mut k = 0;
    while k + 1 < header.len() {
        if is_punct_dot(&header[k]) && is_punct_dot(&header[k + 1]) {
            let mut e = k + 2;
            if e < header.len() && header[e].to_string() == "=" {
                e += 1;
            }
            let t = header.get(e)?;
            return match t {
                TokenTree::Literal(_) | TokenTree::Ident(_) => Some(t.to_string()),
                _ => None,
            };
        }
        k += 1;
    }
    None
}

fn is_punct_dot(t: &TokenTree) -> bool {
    matches!(t, TokenTree::Punct(p) if p.as_char() == '.')
}
