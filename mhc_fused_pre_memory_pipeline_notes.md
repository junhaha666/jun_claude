# MHC fused/pre memory pipeline optimization notes

Date: 2026-06-22

This note records the current optimization state around
`aiter/csrc/kernels/mhc_kernels.cu`, mainly:

- `mhc_fused_post_pre_gemm_sqrsum_kernel`
- `mhc_pre_gemm_sqrsum_kernel`

The main verified improvement so far is on the fused kernel's
`next_residual` write path. The `mhc_pre_gemm_sqrsum_kernel` investigation has
started, but no verified pre-kernel optimization has been kept yet.

## Current Worktree State

Modified files in `/home/jun_chen2_qle/aiter`:

- `aiter/ops/mhc.py`
- `csrc/kernels/mhc_kernels.cu`
- `op_tests/test_mhc.py`

Generated ATT directories:

- `att-logs-pre/`
- `att-logs-fused/`
- `att-logs-fused-direct/`

Important: `op_tests/test_mhc.py` already had local benchmark/reporting edits.
Do not revert it blindly.

## Verified Fused-Kernel Change

Target kernel:

```cpp
mhc_fused_post_pre_gemm_sqrsum_kernel
```

Target case:

```bash
python3 op_tests/test_mhc.py -n 7168 --fuse_rmsnorm -m 4096
```

### Config Change

In `aiter/ops/mhc.py`, for `m=4096, hidden_size=7168`, force:

```python
return 8, 16, 32, 64
```

That means:

```text
split_k = 8
tile_m  = 16
tile_n  = 32
tile_k  = 64
```

This is deliberately narrow. It is meant for the forced-fused `m=4096,
hidden=7168` tuning point.

### Store Path Change

In `mhc_fused_post_pre_gemm_sqrsum_kernel`, `next_residual` store is changed
only for `tile_k == 64`.

Old behavior:

- each MFMA-lane-layout vector directly stores to `g_nres`
- store addresses are not well coalesced

New behavior:

1. Each lane writes its computed `res` vector to LDS:

```cpp
__shared__ DTYPE_I s_nres[(tile_k == 64) ? hc_mult * tile_m * tile_k : 1];
```

2. At the end of the tile, after `__syncthreads()`, lanes reload from LDS in
   row/col order and write `g_nres` more coalesced.

For `tile_k != 64`, the original direct store path is still used.

## Fused Performance Result

Earlier clean baseline for forced fused:

```text
~220.7 us
~2.66 TB/s
```

Verified coalesced-store result range:

```text
~193-196 us
~3.00-3.05 TB/s
```

Approximate improvement:

```text
time reduction:       ~25-28 us
kernel speedup:       ~12-14%
reported BW increase: ~13-15%
```

Later, after repeated profiling/rebuilds, the same source produced slower
absolute timings around `214-215 us`. Since both unfused and fused timings were
slower in that period, treat that as environment/frequency/system-state noise,
not as a source-level regression. The source was restored to the coalesced
version after the temporary direct-store ATT experiment.

## ATT Comparison

The user modified `../inputs.json` to trace:

```json
"kernel_include_regex": "mhc_pre_gemm_sqrsum_kernel"
```

Additional temporary input was generated for:

```text
mhc_fused_post_pre_gemm_sqrsum_kernel
```

### ATT: pre kernel

Directory:

```text
att-logs-pre/
```

Kernel:

```cpp
mhc_pre_gemm_sqrsum_kernel<std::bfloat16_t, 256, 64, 32, 128>
```

Stall summary:

```text
total stall              656,428
buffer_load_dwordx4      272,028  41.4%
s_barrier                121,112  18.4%
s_waitcnt                 57,988   8.8%
ds_read_b128              16,264   2.5%
buffer_store_dwordx4         288   0.0%
```

Interpretation:

- The global reads are mostly contiguous/coalesced, but the kernel is not a pure
  streaming load kernel.
- Significant time is still spent in barriers, waitcnts, LDS reads, bf16 split,
  and MFMA.
- Therefore, "continuous read" alone does not imply HBM peak bandwidth.

### ATT: fused coalesced store

Directory:

```text
att-logs-fused/
```

Kernel:

```cpp
mhc_fused_post_pre_gemm_sqrsum_kernel<std::bfloat16_t, 256, 4, 16, 32, 64, true>
```

Stall summary:

```text
total stall            2,487,348
buffer_load_dwordx4      857,344  34.5%
s_barrier                524,244  21.1%
s_waitcnt                430,640  17.3%
buffer_store_dwordx4     204,808   8.2%
ds_read_b128             123,512   5.0%
global_load_dword        107,408   4.3%
ds_write_b128             16,188   0.7%
```

### ATT: fused direct store

Directory:

```text
att-logs-fused-direct/
```

This was a temporary experiment: the two `if constexpr(tile_k == 64)` branches
were changed to `if constexpr(false)` to force the old direct `g_nres` store
path. The source was restored afterward.

Stall summary:

```text
total stall            2,832,032
buffer_load_dwordx4    1,002,356  35.4%
s_waitcnt                539,128  19.0%
s_barrier                484,172  17.1%
buffer_store_dwordx4     355,396  12.5%
global_load_dword        132,536   4.7%
ds_read_b128              21,912   0.8%
```

### ATT conclusion

Coalesced `next_residual` store is clearly effective:

```text
total stall:             2.832M -> 2.487M   -12.2%
buffer_store_dwordx4:    355K   -> 205K     -42.4%
s_waitcnt:               539K   -> 431K     -20.1%
buffer_load_dwordx4:     1002K  -> 857K     -14.5%
```

Cost introduced by LDS staging:

```text
ds_read_b128:   21.9K -> 123.5K
ds_write_b128:   1.1K -> 16.2K
s_barrier:      484K  -> 524K
```

Net result is positive, but it cannot reach theoretical HBM bandwidth because
the remaining stall is dominated by global loads, waitcnt, barriers, and LDS
pipeline work.

## Experiments Tried And Not Kept

### fused load/store overlap

Idea:

- defer coalesced `next_residual` store
- issue next tile's `fn/x/residual` loads earlier
- then store previous tile

Result:

```text
~195.6 us
```

This was slower than the clean coalesced-store version around `192-193 us`, so
it was not kept.

Likely reason:

- global loads and global stores compete for the same memory pipeline
- deferring store increases synchronization/lifetime pressure
- the next `wait_load_cnt()` still exposes memory dependency

### skip barrier after coalesced store

Idea:

- since `compute_store_tile()` for `tile_k == 64` already has `__syncthreads()`,
  try skipping the following loop-level `s_barrier`

Status:

- not cleanly validated
- not kept

### mem-only diagnostic

Diagnostic experiment with most VALU/MFMA removed while preserving load/store
and sync:

```text
~181.3 us
~3.24 TB/s
```

Conclusion:

- even with compute mostly disabled, the pipeline does not reach 7 TB/s
- bottleneck is not only MFMA/VALU
- memory pipeline shape, waitcnt/barrier, LDS staging, and store layout matter

### config experiments

Observed fused results:

```text
tile_k=64, tile_m=16, split_k around 7/8: best range
tile_k=128:                               ~238.8 us, bad
tile_m=32, tile_k=64:                     ~299 us, bad
tile_n=16, tile_k=64:                     ~280 us, bad
```

`m=1024` and `m=2048` were checked after gating the coalesced store to
`tile_k == 64`; no severe regression was seen in the earlier clean runs:

```text
m=1024: ~56-57 us
m=2048: ~102-104 us
```

## mhc_pre_gemm_sqrsum_kernel Investigation

Target kernel:

```cpp
mhc_pre_gemm_sqrsum_kernel
```

Current ATT instance for `m=4096,n=7168,hc_mult=4`:

```cpp
mhc_pre_gemm_sqrsum_kernel<std::bfloat16_t, 256, 64, 32, 128>
```

Meaning:

```text
block_size = 256
tile_m     = 64
tile_n     = 32
tile_k     = 128
```

The initial pre-kernel optimization direction was to test config changes before
rewriting the loop body, because config affects:

- number of CTAs
- load batch size
- barrier frequency
- split-k occupancy

### Experimental pre config started

An experimental special case was inserted into `get_mhc_pre_splitk`:

```python
if m == 4096 and hc_hidden_size == 28672:
    return 14, 64
```

This tries:

```text
split_k = 14
tile_k  = 64
```

Rationale:

- keep total block count close to the current `tile_k=128, split_k=8` path
- halve per-tile K width
- potentially reduce some long global-load stalls and change wait/barrier shape

Status:

- not benchmarked successfully yet
- Python execution started failing with:

```text
ModuleNotFoundError: No module named 'encodings'
```

and later sandbox/bwrap path errors appeared. The benchmark was interrupted
before this pre-kernel config could be validated.

Important: this pre config is experimental and should either be measured or
removed before committing.

## Immediate Next Steps

1. Recover a working Python benchmark environment.
2. Measure current source twice:

```bash
python3 op_tests/test_mhc.py -n 7168 --fuse_rmsnorm -m 4096
```

3. Specifically compare `mhc_pre` unfused timing for:

```text
baseline: tile_k=128, split_k=8-ish selected by existing heuristic
experiment: tile_k=64, split_k=14
```

4. If `tile_k=64, split_k=14` regresses, remove the special case.
5. If it helps, run ATT for the new pre config and compare:

```text
buffer_load_dwordx4
s_barrier
s_waitcnt
ds_read_b128
v_mfma_f32_16x16x32_bf16
```

6. Only after config sweep, consider deeper pre-kernel loop changes:

- reduce barrier frequency around `GEMM_LOOP_BODY`
- adjust fn LDS staging
- change `tile_k/tile_n` pairing
- reduce `fn` LDS read wait exposure

