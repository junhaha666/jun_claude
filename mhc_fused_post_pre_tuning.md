# Tuning `mhc_fused_post_pre_gemm_sqrsum` for a new chip

This guide explains how to produce a tuned `(split_k, tile_m, tile_n, tile_k)`
config policy for the fused post+pre GEMM kernel on a GPU it has not been tuned
for yet (e.g. a different CU count or `gfx` arch). Follow it end-to-end and you
get a new entry for the per-chip registry in `aiter/aiter/ops/mhc.py`.

---

## 0. Background: what the tunables are

The kernel is `mhc_fused_post_pre_gemm_sqrsum_kernel`
(`csrc/kernels/mhc_kernels.cu`). It is a fused GEMM + residual-mix + sqrsum.
Four knobs control how work is mapped to the device:

| param     | meaning | valid values |
|-----------|---------|--------------|
| `tile_m`  | rows of output per thread-block. `tile_m=32` runs **two mfma m-bands per block reusing one `fn` load** (cuts redundant `fn` traffic). `tile_m=16` is one band. | 16, 32, 64 |
| `tile_n`  | output columns (of `hc_mult3=24`) per block. `tile_n=32` → 1 n-block covers all 24; `tile_n=16` → 2 n-blocks. | 16, 32 |
| `tile_k`  | K-tile (per stream, K = `hidden_size`). Larger = wider loads, fewer loop iters, but more live VGPR / LDS. | 32, 64 |
| `split_k` | how many ways K is split across blocks (each writes a partial, reduced later). More split_k = more blocks to fill the device, but smaller per-block work + extra reduction. | 1 .. num_cu |

**Hard constraints (must hold or the kernel asserts / overflows):**

- `hidden_size % (tile_k * split_k) == 0` and `hidden_size // split_k >= tile_k * 2`
  (2-stage prefetch needs ≥2 k-tiles per block).
- The C++ dispatch only instantiates these **10 combos** (see
  `MHC_FUSED_POST_PRE_GEMM_SQRSUM_KERNEL_DISPATCH`):
  `tile_m ∈ {16,32,64} × tile_n ∈ {16,32} × tile_k ∈ {32,64}`,
  **excluding `tile_m=64 + tile_k=64`** (its `s_residual` LDS = `2*64*hc_mult*64*2B = 64KB`, over budget).
  Requesting any other combo → `TORCH_CHECK(false, ...)`. If you want a combo
  outside this set, add a `MHC_FUSED_POST_PRE_GEMM_SQRSUM_KERNEL_CASE(...)` line
  and rebuild.
- `hc_mult == 4` is assumed, so `hc_mult3 = hc_mult*2 + hc_mult^2 = 24` always.

**Known facts from the gfx950/256 tuning (don't re-derive these blind):**

- `tile_m=64` was **~2x slower** even at full occupancy (VGPR spill: 4 m-bands of
  `v_cf` + res state). Treat `tile_m=64` as a last resort; expect it to lose.
- `tile_m=32` only pays off when the grid still fills the device; on **low-CU
  parts it regressed**, which is exactly why per-chip tuning exists.
- `tile_k=64 + tile_m=32` is impossible (LDS). So "wide-k" and "fn-reuse" can't be
  combined — that tradeoff is the heart of the tuning.
- `tile_n=16` (2 n-blocks) **never won** on gfx950/256. It looks like it should help
  occupancy at small m, but it lost everywhere — badly at small m (m=128:
  14.8us @ tn=16 vs 10.4us @ tn=32, **1.4x**). The policy hardcodes `tile_n=32`.
  Re-measure on your chip before trusting tn=16, but expect the same.
- The `tile_k` crossover is **hidden_size-dependent**, not a pure function of `m`.
  On gfx950/256 it sat at `m >= 2*hidden_size` (h=7168 crossed at m=16384, h=4096
  at m=8192). A crossover written only in terms of `m`/`num_cu` will mis-fire when
  `hidden_size` changes.
- On the compute-bound `tile_k=64` path, the geometric split_k fill returns
  `split_k=1` at large m (ideal `32*cu/m ≈ 1`), but `split_k=2` measured ~2-5%
  faster (one extra K-reduction wave). The policy floors split_k at 2 on that
  branch when 2 is a legal divisor.

---

## 1. Set up the bench harness

The test column `fused_us` is **end-to-end** (this kernel + the second
`mhc_pre_big_fuse` kernel + launch overhead). To tune **this kernel alone**, call
the op directly and time it with CUDA events. Save this as `/tmp/probe_mhc.py`:

The harness takes `(m, hidden)` cases. **Sweep at least two `hidden` values** — the
`tile_k` crossover depends on `hidden_size`, so a single-`hidden` sweep can't pin it.

```python
import torch, sys, math
import aiter
from aiter import dtypes
torch.set_default_device("cuda")

hc_mult = 4
hc_mult3 = hc_mult*2 + hc_mult*hc_mult   # 24

def make_inputs(m, splitk, hidden):
    n = splitk*hc_mult
    pad = (hc_mult3+31)//32*32
    go = torch.empty(n, m, pad, dtype=dtypes.fp32)[:, :, :hc_mult3]
    gs = torch.empty(n, m, dtype=dtypes.fp32)
    nr = torch.empty(m, hc_mult, hidden, dtype=dtypes.bf16)
    li = torch.randn(m, hidden, dtype=dtypes.bf16)
    ri = torch.randn(m, hc_mult, hidden, dtype=dtypes.bf16)
    pm = torch.randn(m, hc_mult, dtype=dtypes.fp32)
    cm = torch.randn(m, hc_mult, hc_mult, dtype=dtypes.fp32)
    fn = torch.randn(hc_mult3, hc_mult*hidden, dtype=dtypes.fp32)
    return (go, gs, nr, li, ri, pm, cm, fn)

def valid(hidden, tk, sk):
    return hidden % (tk*sk) == 0 and hidden//sk >= tk*2

def bench(m, sk, tm, tn, tk, hidden, iters=50):
    if not valid(hidden, tk, sk):
        return None
    a = make_inputs(m, sk, hidden)
    f = lambda: aiter.mhc_fused_post_pre_gemm_sqrsum(*a, tm, tn, tk)
    try:
        for _ in range(10): f()      # warmup; also catches unsupported combos
    except Exception:
        return None
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters): f()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e)/iters*1000   # microseconds

# the 10 valid (tile_m, tile_n, tile_k) combos
COMBOS = [(tm,tn,tk) for tm in (16,32,64) for tn in (16,32) for tk in (32,64)
          if not (tm==64 and tk==64)]

# args are "m,hidden" pairs; default sweeps two hidden sizes to locate the tile_k crossover
CASES = [(m,h) for h in (7168,4096)
         for m in (64,128,256,512,1024,2048,4096,8192,16384)]
if sys.argv[1:]:
    CASES = [tuple(int(x) for x in a.split(",")) for a in sys.argv[1:]]

for (m,hidden) in CASES:
    results = []
    print(f"\n=== m={m} hidden={hidden} ===", flush=True)
    for (tm,tn,tk) in COMBOS:
        for sk in [x for x in range(1, 257) if valid(hidden, tk, x)]:
            t = bench(m, sk, tm, tn, tk, hidden)
            if t is not None:
                results.append((t, sk, tm, tn, tk))
    results.sort()
    for t, sk, tm, tn, tk in results[:8]:   # top-8 so you can see ties / 2nd-best
        print(f"  splitk={sk:3d} tile_m={tm} tile_n={tn} tile_k={tk}  ->  {t:.1f} us", flush=True)
    t, sk, tm, tn, tk = results[0]
    print(f"  >>> BEST: splitk={sk} tile_m={tm} tile_n={tn} tile_k={tk}  ->  {t:.1f} us", flush=True)
```

Run it on the target GPU (whole default sweep, or specific cases):

```bash
cd /home/jun_chen2_qle/aiter
python3 /tmp/probe_mhc.py 2>&1 | grep -vE "^\[aiter|Warning|import|build"   # full sweep
python3 /tmp/probe_mhc.py 128,7168 8192,4096 2>&1 | grep -E "===|splitk|BEST" # targeted
```

> The full sweep is large (10 combos × ~10 split_k × #cases). To iterate fast, pass
> specific `m,hidden` cases as args and read the top-8 to spot near-ties (anything
> within ~2-3% is noise — prefer the simpler combo).

---

## 2. Read the results — what to look for

For each `m`, you get the empirically fastest `(split_k, tile_m, tile_n, tile_k)`.
Tabulate them. Patterns that showed up on gfx950/256 (yours may differ — that's
the point of tuning):

- **tile_k**: 32 wins until `m` is large enough to be compute-bound, then 64. The
  crossover is **hidden_size-dependent** — on gfx950/256 it was `m >= 2*hidden_size`
  (NOT a pure `m`/`num_cu` rule; a function of `m` alone mis-fires when
  `hidden_size` changes). Sweep at ≥2 different `hidden_size` values to pin where
  the crossover lands as a function of both.
- **tile_m**: 32 wins whenever the grid still fills the device (`ceil(m/32)*split_k
  >= num_cu`); else 16. On low-CU chips 32 may never win — then keep 16 everywhere.
  `tile_m=32` is tied to the `tile_k=32` path (LDS forbids `tile_k=64 + tile_m=32`).
- **tile_n**: on gfx950/256, **always 32** — `tile_n=16` never won despite the
  occupancy intuition. Measure both, but if `tile_n=16` doesn't clearly win, drop
  it and hardcode 32 (simpler and avoids a small-m regression).
- **split_k**: the optimum sat near `total_blocks ≈ 32*num_cu`, i.e.
  `split_k ≈ 32*num_cu/m`, snapped to a valid divisor. Verify this ratio holds on
  your chip; the constant (32) may shift with CU count / memory system. **Caveat:**
  on the compute-bound `tile_k=64` path the fill returns `split_k=1` at large m,
  but `split_k=2` was ~2-5% faster — floor it at 2 there if your sweep agrees.

**Sanity rule:** time differences under ~2–3% are noise. Don't encode a combo
switch that only buys 1%. Prefer the simpler/more-uniform choice.

---

## 3. Turn the table into a policy function

A policy is just `fn(m, hidden_size, num_cu) -> (split_k, tile_m, tile_n, tile_k)`.
Two ways to write it, pick whichever fits your data:

**(a) Analytic** — if the optima follow clean rules (like gfx950 did). Mirror the
existing `_mhc_fused_config_gfx950_256` in `aiter/ops/mhc.py`:

```python
def _mhc_fused_config_<arch>_<cu>(m, hidden_size, num_cu):
    tile_k = 64 if m >= 2*hidden_size else 32             # <- your measured crossover
    if hidden_size % tile_k != 0:
        tile_k = 32 if tile_k == 64 else 64
    valid = _mhc_fused_valid_splitk(hidden_size, tile_k, num_cu)
    if not valid:
        return 1, 16, 32, tile_k
    splitk = _mhc_fused_fill_splitk(m, valid, num_cu)      # geometric snap to 32*cu/m

    tile_n = 32                                           # tn=16 never won; verify on your chip
    if tile_k == 32:
        tile_m = 32 if (m + 31)//32 * splitk >= num_cu else 16   # fn-reuse when grid fills
    else:  # tile_k == 64 forces tile_m=16 (LDS); floor split_k at 2 (compute-bound)
        tile_m = 16
        if splitk < 2 and 2 in valid:
            splitk = 2
    return splitk, tile_m, tile_n, tile_k
```

This is the exact shape of the current `_mhc_fused_config_gfx950_256`. The policy is
**fully analytic** (every knob is a formula in `m`/`hidden_size`/`num_cu`), so any
`m` — including non-power-of-two — resolves to a config. There is **no pad-m / pow2
bucketing** anywhere, and none is needed; don't add one unless your measured optima
are jagged in `m` (then use the lookup table form below).

Reuse the helpers already in `mhc.py`:
- `_mhc_fused_valid_splitk(hidden_size, tile_k, num_cu)` → list of legal split_k.
- `_mhc_fused_fill_splitk(m, valid, num_cu)` → split_k nearest (geometric) to `32*num_cu/m`.

**(b) Lookup table** — if the optima are irregular, hardcode them. Bucket by `m`
(use ranges, not exact values, so unseen `m` still resolves):

```python
def _mhc_fused_config_<arch>_<cu>(m, hidden_size, num_cu):
    # (m_upper_bound, (splitk, tile_m, tile_n, tile_k))
    TABLE = [
        (64,    (..., 16, 16, 32)),
        (512,   (..., 32, 16, 32)),
        (8192,  (..., 32, 32, 32)),
        (1<<30, (...,  16, 32, 64)),
    ]
    for ub, cfg in TABLE:
        if m <= ub:
            # recompute split_k from cfg's tile_k so it stays legal for this hidden_size
            sk_fixed, tm, tn, tk = cfg
            valid = _mhc_fused_valid_splitk(hidden_size, tk, num_cu)
            sk = min(valid, key=lambda s: abs(s - sk_fixed)) if valid else 1
            return sk, tm, tn, tk
    ...
```

> Always recompute / clamp `split_k` against `_mhc_fused_valid_splitk` for the
> *actual* `hidden_size` at call time — a split_k hardcoded for n=7168 may be
> illegal for another hidden size.

---

## 4. Register the policy

Add one line to the registry in `aiter/ops/mhc.py`, keyed by `(gfx_arch, cu_num)`:

```python
_MHC_FUSED_POST_PRE_CONFIG = {
    ("gfx950", 256): _mhc_fused_config_gfx950_256,
    ("gfx942", 304): _mhc_fused_config_gfx942_304,   # <- your new entry
}
```

- `gfx_arch` comes from `get_gfx_runtime()`, `cu_num` from `get_cu_num()`.
- Chips with no entry fall back to `_mhc_fused_config_default` (safe: tile_m=16 +
  occupancy-scoring split_k/tile_k search). So adding an entry is purely additive
  and can't break other chips.

No C++ rebuild is needed unless you added a new dispatch CASE in step 0.

---

## 5. Verify

1. **Config routing** — confirm your policy is selected and emits legal combos:
   ```bash
   python3 -c "
   from aiter.ops.mhc import get_mhc_fused_post_pre_config
   for m in [1,64,256,1024,2048,8192,16384]:
       print(m, get_mhc_fused_post_pre_config(m, 7168))
   "
   ```
   Every tuple must be one of the 10 valid combos.

2. **Correctness** — full sweep must pass (the `fused/*` rows; a small
   `warning!` on `layer_input`/`comb_mix` at ~0.01–0.03 is the expected baseline
   pattern, not a failure):
   ```bash
   python3 op_tests/test_mhc.py -n 7168
   ```

3. **Speedup** — re-run `/tmp/probe_mhc.py` and confirm your policy's choice is
   within a few % of the per-m BEST it reports. If a particular `m` is far off,
   fix that bucket/branch in the policy.

---

## Appendix: gfx950 / 256 CU reference numbers

Per-`(m, hidden)` optimum found by the sweep (this kernel timed alone, CUDA
events). These are what the current `_mhc_fused_config_gfx950_256` emits.

hidden = 7168:

| m     | best (splitk, tm, tn, tk) | note |
|-------|---------------------------|------|
| 16    | (112, 16, 32, 32)         | launch-bound (~8us), config barely matters |
| 64    | (112, 16, 32, 32)         | tn=32 beats old tn=16 pick (~1.1x) |
| 128   | (56,  16, 32, 32)         | tn=32 beats old tn=16 pick (**1.42x**) |
| 256   | (32,  32, 32, 32)         | 1.36x vs legacy always-tk64 |
| 512   | (16,  32, 32, 32)         | 1.10x |
| 1024  | (8,   32, 32, 32)         | 1.13x |
| 2048  | (4,   32, 32, 32)         | 1.14x |
| 4096  | (2,   32, 32, 32)         | 1.12x |
| 8192  | (1,   32, 32, 32)         | 1.05x |
| 16384 | (2,   16, 32, 64)         | tk crossover (m >= 2*hidden); sk=2 > sk=1 |

hidden = 4096 (crossover lands earlier because the rule is `m >= 2*hidden`):

| m     | best (splitk, tm, tn, tk) | note |
|-------|---------------------------|------|
| 2048  | (4,   32, 32, 32)         | tile_k=32 path |
| 4096  | (2,   32, 32, 32)         | tile_k=32 path |
| 8192  | (2,   16, 32, 64)         | tk crossover at m=2*4096; **1.14x** vs old tk=32 |
| 16384 | (2,   16, 32, 64)         | sk=2 > sk=1 (~1.03x) |

Key takeaways that generalize:
- **tile_m=32 (fn-reuse) beats wide tile_k for most m on a high-CU part**, but the
  win shrinks as m grows and flips to tile_k=64 once the kernel is compute-bound.
  On low-CU parts the fn-reuse path can't fill the device and regresses — re-measure.
- The **tile_k crossover scales with hidden_size** (`m >= 2*hidden_size` here), not
  with m alone.
- **tile_n=16 never helps** on this chip — don't pay its small-m cost.
- On the tile_k=64 branch, **floor split_k at 2** (the geometric fill underfills to 1).
