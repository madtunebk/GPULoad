#!/usr/bin/env python3
"""
Patch candle flux model.rs at the pinned git rev to expose private items
required by the StreamForge Rust binaries.

Safe to run multiple times (idempotent — replacements only apply when
the original private signature is still present).

Usage:
    python3 scripts/patch_candle.py
"""
import glob, sys

CANDLE_REV = "d23664f"

matches = glob.glob(
    f"/root/.cargo/git/checkouts/candle-*/{CANDLE_REV}"
    f"/candle-transformers/src/models/flux/model.rs"
)
if not matches:
    print(f"ERROR: candle model.rs not found for rev {CANDLE_REV}", file=sys.stderr)
    sys.exit(1)

path = matches[0]
src = open(path).read()

# Each tuple: (old_string, new_string)
# Strings include enough context to be unambiguous and to avoid accidentally
# adding `pub` to trait-impl items (which Rust forbids: E0449).
#
# Specifically, Flux's WithForward trait impl has:
#   fn forward(&self, img: &Tensor, img_ids: &Tensor, ...)  ← second arg is img_ids
# DoubleStreamBlock's impl block has:
#   fn forward(&self, img: &Tensor, txt: &Tensor, ...)      ← second arg is txt
# We include the second arg to distinguish them.
replacements = [
    # timestep_embedding: free function, pub(crate) in original commit
    (
        'pub(crate) fn timestep_embedding',
        'pub fn timestep_embedding',
    ),
    # DoubleStreamBlock::new and SingleStreamBlock::new both start with
    # `let h_sz = cfg.hidden_size;` — replace all occurrences (both need pub)
    (
        '    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {\n'
        '        let h_sz = cfg.hidden_size;',
        '    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {\n'
        '        let h_sz = cfg.hidden_size;',
    ),
    # DoubleStreamBlock::forward — second arg is `txt: &Tensor`
    (
        '    fn forward(\n'
        '        &self,\n'
        '        img: &Tensor,\n'
        '        txt: &Tensor,',
        '    pub fn forward(\n'
        '        &self,\n'
        '        img: &Tensor,\n'
        '        txt: &Tensor,',
    ),
    # SingleStreamBlock::forward — first arg is `xs: &Tensor`
    (
        '    fn forward(&self, xs: &Tensor, vec_: &Tensor, pe: &Tensor) -> Result<Tensor> {',
        '    pub fn forward(&self, xs: &Tensor, vec_: &Tensor, pe: &Tensor) -> Result<Tensor> {',
    ),
]

patched = src
applied = []
for old, new in replacements:
    count = patched.count(old)
    if count > 0:
        patched = patched.replace(old, new)
        applied.append((old.split('\n')[0].strip(), count))

if patched != src:
    open(path, 'w').write(patched)
    print(f"Patched {path}")
    for sig, n in applied:
        plural = f" ({n}x)" if n > 1 else ""
        print(f"  • {sig}{plural}")
else:
    print("No changes — already patched or signatures not found")
