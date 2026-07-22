# DeepSeek-V4 FP4 Indexer Integration

How the FlyDSL FP4 paged-MQA-logits kernels are wired into DeepSeek-V4's CSA
Indexer in ATOM, as an alternative to the default FP8 index-scoring path.

## Background

The V4 CSA Indexer selects the top-k compressed KV positions per query via a
learned score. The default path scores in FP8:

- prefill: `cp_gather` + `fp8_mqa_logits`
- decode:  `deepgemm_fp8_paged_mqa_logits`

The FP4 pass runs the same scoring fully in FP4 (Q FP4 × KV FP4) using two
gfx950 FlyDSL kernels that read the paged FP4 indexer cache directly via
`block_tables` (no gather):

- prefill: `flydsl_pa_mqa_logits_fp4_prefill` (ragged)
- decode:  `flydsl_pa_mqa_logits_fp4` (varctx)

Both emit **seq-local** logits consumed by `top_k_per_row_{prefill,decode}`.

## Enabling

The FP4 indexer is a first-class CLI flag:

```
--enable-deepseek-v4-fp4-indexer
```

Flow: `EngineArgs.enable_deepseek_v4_fp4_indexer` → engine kwargs →
`Config.enable_deepseek_v4_fp4_indexer`. The attention metadata builder reads
`model_runner.config.enable_deepseek_v4_fp4_indexer` at init to size the cache
pools; the runtime scoring path then auto-detects FP4 via
`kv_cache.dtype == uint8` (the FP4 indexer cache is two uint8 pools; the FP8
cache is `torch.float8`). When the flag is off, the FP8 path is byte-for-byte
unchanged.

- **gfx950-only** (the kernels use `mfma_scale_f32_16x16x128_f8f6f4`). The flag
  is honored only on gfx950; on any other arch the builder logs a warning and
  falls back to the FP8 indexer (no hard error).
- The flag is part of the config hash (`Config.compute_hash` factors), so
  toggling it selects a distinct compile/cache key.
- The legacy `ATOM_V4_INDEXER_FP4` env var has been **removed**; use the flag.

Example:

```bash
python -m atom.entrypoints.openai.api_server --model /data/DeepSeek-V4-Pro -tp 8 \
    --enable-deepseek-v4-fp4-indexer
```

## block_size = 256 (required)

V4's classical KV `block_size` must be a multiple of `lcm(m, m') = lcm(4, 128)
= 128`. It is set to **256** (= 2·lcm) so each block holds:

- `k1_csa = block_size // 4 = 64` CSA entries
- `k2_hca = block_size // 128 = 2` HCA entries

`k1_csa = 64` is the load-bearing requirement: it becomes the FP4 indexer cache
`kv_block_size`, and the FP4 mqa-logits kernels require `kv_block_size == 64`
(see "Scale interleave" below). `block_size` is a single global value enforced
in three places that MUST agree:

| Location | Constant |
| --- | --- |
| `atom/config.py` (`v4_block_size`) | forces `config.kv_cache_block_size = 256` |
| `atom/model_ops/attentions/deepseek_v4_attn.py` (`DeepseekV4AttentionMetadataBuilder.block_size`) | `256` |
| `atom/models/deepseek_v4.py` (`_V4_BLOCK_SIZE`) | `256` |

Density is conserved (block 2× larger, `num_blocks` halved); the only cost is
slightly coarser paging (tail-block fragmentation). Prefix caching is already
disabled for V4 (SWA buffer not cacheable), so block size does not affect it.

## FP4 cache layout

The FP4 indexer cache is two uint8 pools (vs the FP8 path's single fp8 + fp32
strided-scale pool), allocated when the flag is on:

```
v4_csa_idx_kv        : [n_csa, num_blocks, k_tiles, 4, k1_csa, 16] uint8  (packed E2M1 data)
v4_csa_idx_kv_scale  : [n_csa, num_blocks, k_tiles, 4, k1_csa]     uint8  (e8m0 MX block scale)
```

`k_tiles = index_head_dim // 128` (= 1 for D=128). E2M1: 2 fp4 per byte;
group_size = 32 with one e8m0 byte per group.

### Scale interleave (writer ↔ reader contract)

The e8m0 scale slot axis is **interleaved** so the reader's packed-dword load
(the NTPW nt-bytes of a token adjacent) is contiguous:

```
sflat = (slot_in_block % 16) * (kv_block_size / 16) + (slot_in_block // 16)
```

For the indexer's `kv_block_size == 64` this is `(slot%16)*4 + slot//16`
(`KVS_NTPW = 4`). Data is linear by `slot_in_block` (`[..., slot, 16]`); only
the scale axis is interleaved. This single formula is shared by:

- the production writer `fused_compress_attn` (`quant_mode="fp4"`)
- `dsv4_rotate_quant.cu` (`kv_scale_preshuffle_offset`, the standalone
  rmsnorm+rope+fp4 KV writer)
- the torch reference `aiter/ops/torch_ref/fused_compress_attn.py`
- the op-test reference writers
- the packed-dword readers in `pa_mqa_logits_fp4{,_prefill}`

The readers assume `N_PHYS == 1` (the NTPW N-tiles of a warp share one physical
block), which holds iff `kv_block_size == 64` (`TILES_PER_BLOCK = 64/16 = 4 =
NTPW`). ATOM asserts `kv_block_size == 64` in both FP4 score paths so an
unsupported block size fails loudly instead of misreading scales.

## Forward wiring

### Q FP4 quant (`Indexer.forward_batched`)

When `kv_cache.dtype == uint8`:

- `rope_rotate_activation(out=fp4x2, scale=q_scale, group_size=32,
  shuffle_scale=True)` produces packed `q_fp4` + preshuffled e8m0 `q_scale`.
- weights = `weights_proj(x)` (bf16) passed straight to the mqa kernel, which
  loads weights as bf16 and applies the static `_weights_scale` internally via
  its `weight_scale` arg. No float cast, no per-row q_scale premultiply (the mqa
  kernel dequants Q via e8m0), no pre-scale launch. This is leaner than the FP8
  path (which also runs `scale_indexer_weights`).
- `q_scale` is stashed on `self._q_scale_fp4` for the opaque dispatch op.

### Compressor (FP4 KV write)

`Compressor.forward` selects `quant_mode`:

```
"fp4" if (is_quant and kv_cache.dtype == uint8) else ("fp8" if is_quant else "none")
```

and calls `fused_compress_attn(quant_mode="fp4", cache_scale=<uint8 e8m0 pool>)`,
which writes the data + interleaved-scale layout above.

### Scoring dispatch (`indexer_score_topk`)

Auto-selects FP4 when `kv_cache.dtype == uint8`, else the existing FP8 path:

- `_score_topk_prefill_fp4` → `flydsl_pa_mqa_logits_fp4_prefill` →
  `top_k_per_row_prefill`. Logits are seq-local, so indices are returned
  directly (no `seq_base` subtract, unlike the FP8 GLOBAL output).
- `_score_topk_decode_fp4` → `flydsl_pa_mqa_logits_fp4` → `top_k_per_row_decode`.

## Persistent-grid schedule (cta_info)

The FP4 mqa-logits kernels use a persistent grid: a fixed number of CTAs
(`parallel_unit_num`) each handle a precomputed `(row/batch, chunk-split)` work
slot listed in a `cta_info` table. The table is built **before** the kernel so
the launch is a pure grid dispatch (CUDAGraph-friendly).

`cta_info` is built by a **single fused Triton kernel** per fwd:

- decode: `compute_varctx_schedule` (in `aiter/.../pa_mqa_logits_fp4.py`)
- prefill: `compute_prefill_schedule` (in `.../pa_mqa_logits_fp4_prefill.py`)

Earlier these were ~50 tiny torch ops each (arange / broadcast-divide / sum /
cumsum / searchsorted / gather / stack). Being launch-bound, that cost
~300 µs/call **regardless of problem size**, and decode rebuilds it every token,
so it dominated the FP4 path (mqa kernel faster than FP8 but end-to-end slower).
Fusing to one kernel:

| schedule | before | after |
| --- | --- | --- |
| decode (`compute_varctx_schedule`) | ~321 µs | ~48 µs |
| prefill (`compute_prefill_schedule`) | ~300 µs | ~174 µs |

Both are bit-exact with the retained torch references (`_compute_*_schedule_torch`).
Prefill keeps its (sync-free) torch `safe`-search + cumsum and fuses only the
per-slot mapping+emit, because `total_tokens` can be ~16k (too large for the
per-program prefix recompute the decode kernel uses).

### Schedule sizing (`parallel_unit_num`)

`parallel_unit_num` (= persistent-grid CTA count) caps two axes:

- **rows**: every `(row, chunk-split)` needs a slot; a prefill fwd has one row
  **per query token** and a decode fwd one per decode token, so `P` must be
  `>= rows` or the surplus is silently dropped (→ NaN logits → wrong top-k).
- **chunks**: a floor keeps enough CTAs to split a long context across the GPU
  when the batch is small.

ATOM therefore sizes `P = max(_FP4_DECODE_PARALLEL_UNIT_NUM(512), rows)`:

- prefill: `max(512, prefill_rows)` in `_build_v4_indexer_meta`.
- decode: `max(512, T_dec)` (`T_dec = max_bs * (1 + max_spec_steps)`); the fixed
  CG `cta_info` buffer is sized to the same `P`.

`compute_{varctx,prefill}_schedule` assert `P >= rows` (host-side, sync-free) so
a too-small grid fails loudly instead of silently dropping work.

Shared params (must match builder and score call):

```
_FP4_DECODE_PARALLEL_UNIT_NUM = 512   (P floor / total_ctas)
_FP4_DECODE_BLOCK_K           = 256
```

### CUDAGraph

- **Decode**: the fused schedule kernel writes straight into a fixed-address
  buffer (`self._v4_fp4_cta_info`, `[P, 4]`) via `cta_info_out=`, no intermediate
  alloc or `.copy_`. `total_ctas == P` is constant → the captured grid is stable;
  only buffer contents change per fwd → CUDAGraph-safe.
- **Prefill**: per-fwd tensor (prefill is eager; dynamic total_tokens).

### int64 output addressing

The mqa-logits kernels compute the output store offset as
`row * stride_out + token`. `stride_out = max_model_len // 4` can be large, so
`row * stride_out` overflowed **int32** past `~2^31 / stride`, wrapping the
offset and silently dropping rows (their logits stay at the pre-fill → wrong
top-k). Fixed by folding the per-row base into the buffer resource's 64-bit base
pointer, leaving only the small per-token offset on the 32-bit voffset. Applied
to both `pa_mqa_logits_fp4` and `pa_mqa_logits_fp4_prefill`.

### Logits buffer (no -inf fill)

Both paths allocate logits with `torch.empty` and pass `out=`. The kernel writes
every column in each row's valid window (`[0, visible_end)` prefill /
`[0, context_len)` decode), and the top-k scans only that same window — cells
past it are never read. Skipping the kernel's default `torch.full(-inf)` removes
a large `FillFunc` (width `max_model_len_idx`) before each launch.

## KV transfer (disaggregated serving)

The KV-transfer path raises `NotImplementedError` only when disagg is actually
enabled (`config.kv_transfer_config` non-empty) AND the FP4 indexer is on — the
FP4 scale pool is not yet described by the transfer region map. Single-node
(no disagg) runs normally.

## Files

ATOM:

- `atom/config.py` — `v4_block_size = 256`; `enable_deepseek_v4_fp4_indexer`
  field + config-hash factor
- `atom/model_engine/arg_utils.py` — `--enable-deepseek-v4-fp4-indexer` CLI + arg
- `atom/model_ops/attentions/deepseek_v4_attn.py` — `block_size=256`, FP4 cache
  alloc + binding, gfx950 warn-and-fallback, fused cta_info schedule build
  (fixed CG buffer), `parallel_unit_num` sizing, KV-transfer guard
- `atom/models/deepseek_v4.py` — `_V4_BLOCK_SIZE=256`, Q FP4 quant, Compressor
  `quant_mode`, FP4 score dispatch + `kv_block_size==64` assert
- `atom/model_ops/v4_kernels/fused_compress.py` — `quant_mode` plumbing

aiter:

- `aiter/ops/flydsl/kernels/pa_mqa_logits_fp4{,_prefill}.py` — FP4 mqa-logits
  readers (packed-dword interleaved scale, `N_PHYS==1`); int64 output addressing;
  fused single-kernel `compute_{varctx,prefill}_schedule` + `P >= rows` guard
- `aiter/ops/flydsl/kernels/fused_compress_attn.py` — FP4 KV writer
  (`quant_mode="fp4"`, interleaved scale); single `quant_mode` selector
  (`none|per_row_fp8|group_fp8|fp4`). Both the legacy single-wave and K-split
  kernels emit the FP4 scatter; the FP4 indexer shape auto-engages K-split at
  decode/small-batch and falls back to legacy at high N.
- `aiter/ops/torch_ref/fused_compress_attn.py` — FP4 reference
- `aiter/csrc/kernels/dsv4_rotate_quant.cu` — `rmsnorm_rope_rotate_activation_fp4quant_kvcache`;
  `kv_scale_preshuffle_offset` uses the interleaved scale layout

## Constraints

- gfx950 only (warn + FP8 fallback elsewhere).
- `block_size == 256` (`k1_csa == 64`); asserted in the FP4 score paths.
- `parallel_unit_num >= rows` (asserted); ATOM sizes it as `max(512, rows)`.
- Decode FP4 is CUDAGraph-capturable; prefill FP4 is eager (dynamic total_tokens).
- KV transfer + FP4 indexer not supported together (guarded).
