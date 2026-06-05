# mhc_fused_post_pre_gemm_sqrsum — FMA 重写实现笔记

> MI355X (gfx950, 256 CU) 上对 `mhc_fused_post_pre_gemm_sqrsum_kernel` 的 packed-FMA 重写。
> 测例:`m=2048, hidden_size=7168, hc_mult=4`。
> 源文件:`aiter/csrc/kernels/mhc_kernels.cu`(新 kernel 名 `mhc_fused_post_pre_gemm_sqrsum_fma_kernel`)。

---

## 1. 这个 kernel 算什么

每个 token row `m`、每个 hc 流 `warp_id ∈ [0,4)`,沿 `k ∈ [0, hidden_size)` 做两件事:

1. **post 融合(逐元素)**,产出 `next_residual`:
   ```
   res[m,k] = x[m,k] * post_mix[m,warp]
            + Σ_{h=0..3} residual[m,h,k] * comb_mix[m,h,warp]
   ```
2. **pre GEMM + 平方和**,基于上面的 `res`:
   ```
   out[m, n]  = Σ_k  fn[n, warp*hidden + k] * res[m,k]      (n ∈ [0,24))
   sqrsum[m]  = Σ_k  res[m,k]^2
   ```

输入/输出张量:

| 名称 | 形状 | 说明 |
|---|---|---|
| `x` (layer_input) | `(m, hidden)` bf16 | attn/ffn 输出 |
| `residual` | `(m, hc_mult, hidden)` bf16 | 残差流 |
| `post_layer_mix` | `(m, hc_mult)` fp32 | post 标量 |
| `comb_res_mix` | `(m, hc_mult, hc_mult)` fp32 | 组合标量 |
| `fn` | `(hc_mult3=24, hc_mult*hidden)` fp32 | GEMM 权重 |
| `next_residual` | `(m, hc_mult, hidden)` bf16 | **输出** = `res` |
| `out` (gemm_out_mul) | `(hc_mult, m, 24)` fp32 | **输出** GEMM |
| `sqrsum` | `(hc_mult, m)` fp32 | **输出** 平方和 |

`hc_mult=4`, `hc_mult3 = hc_mult² + 2·hc_mult = 24`。

---

## 2. 为什么用 FMA 重写(动机)

原 mfma 版用 `v_mfma_f32_16x16x4f32`,N 维必须凑 16 的倍数。`hc_mult3=24`
被 `tile_n=16` 切成 **2 个 n-block**(24→32,25% 浪费),且 grid 需要填满 256 CU
→ `m_blocks = m/16 = 128`,只有靠 `n_blocks=2` 才凑到 256 个 block。

代价是 `n_blocks=2` 把 `x` / `residual` 各读两遍、`next_residual` 写两遍:

- 唯一 HBM 流量 ~268 MB,但实际发起 ~617 MB(2× 冗余)。

**FMA 思路(用户提出):** packed `v_pk_fma_f32` 与 mfma 同吞吐
(都是 32 MAC/cyc/SIMD),但不受 N=16 约束 → 可以
**`tile_m=8` + `n_blocks=1`(一个 block 算满 24 列)**:

- `n_blocks=1` → `x`/`residual`/`next_residual` 只读写一遍(617→~340 MB)。
- `tile_m=8` → `m_blocks = m/8 = 256` 重新填满 256 CU。
- 顺带消掉 N 维 24→32 padding 浪费。

---

## 3. 线程 / 数据布局

- `block_size = 256 = hc_mult(4) × warp(64)`。**每个 warp 负责一个 hc 流**(`warp_id`)。
- 一个 block 处理 `tile_m=8` 行 token、全部 `tile_n=24` 列、沿 K 分 `tile_k=32` 的 tile 迭代。
- **warp 内 64 lane 的二维布局**:
  ```
  m_lane = lane % tile_m        // ∈ [0,8)   该 lane 负责哪一行
  kg     = lane / tile_m        // ∈ [0,8)   k-group 编号
  k_base = kg * vec_tile        // 该 lane 在 tile 内的 k 起点
  vec_tile = tile_k / (warp/tile_m) = 32 / 8 = 4   // 每 lane 持有 4 个连续 k
  ```
  即 8 行 × 8 个 k-group,每 k-group 4 个 k,刚好覆盖 `tile_m×tile_k = 8×32`。

- **LDS 三块,均双缓冲**(`n_stages=2`,`(k&1)` 选 slot):
  ```cpp
  __shared__ DTYPE_I s_x       [2 * tile_m * tile_k];              // x tile
  __shared__ DTYPE_I s_residual[2 * hc_mult * tile_m * tile_k];    // 4 个残差流
  __shared__ float   s_fn      [2 * hc_mult * tile_n * tile_k];    // fn,每 warp 一段
  ```

- **跨 k-group 归约**:`res^2` 与 GEMM 累加都是每个 lane 持有部分和(只含自己的
  4 个 k),最后要把同一 `m_lane` 的 8 个 kg lane 求和。用 `ds_bpermute` 蝶形归约:
  ```cpp
  template <int tile_m>
  __device__ float cross_kg_sum(float val, int lane_id) {
      for (int s = tile_m; s < 64; s <<= 1)        // tile_m=8: s=8,16,32
          val += bpermute(lane ^ s, val);
      return val;                                   // 结果落在 kg==0 的 lane
  }
  ```

---

## 4. Pipeline 结构

整体是 **2-stage 软件流水 + unroll-by-2**,从原 mfma kernel 移植过来。三条
load 流(x / residual / fn)全部走 `async_load`(buffer_load → LDS),`s_fn`
也异步进 LDS(原 mfma 把 fn 放 VGPR)。

```
prologue:  issue_tile(0); issue_tile(1)          // 预取 2 个 tile
                                                  // issue = x+residual+fn 三条 async load

主循环 (unroll 2, i += 2):
    wait_load_cnt()         ─┐ 两段 vmcnt 阈值,中间夹一个 s_barrier
    compute_store_tile(i)    │ 消费 slot i&1
    s_barrier                │
    issue_tile(i+2)          │ 预取下一 tile 到 slot (i+2)&1 = i&1(WAR 靠 barrier 保护)
    sched_barrier(0)         │
    wait_load_cnt()          │
    compute_store_tile(i+1)  │
    s_barrier                │
    issue_tile(i+3)         ─┘

epilogue:  处理 k_loop 末尾的 1~2 个残余 tile(带 vmcnt(0) 收尾)

收尾:
    sqrsum:  cross_kg_sum<tile_m> 归约 → kg==0 lane 写出
    out:     每个 n 先 acc[n][0]+acc[n][1],再 cross_kg_sum → kg==0 lane 写出
```

### `wait_load_cnt()` —— 手调 vmcnt 阈值

在途 vmem op 数随线程身份不同(`x` 只有前 `x_async_load_threads` 个线程参与):

```cpp
// 第一段:留下「1 个 tile 的 fn + 当前 tile」在途,过 barrier
vmcnt(x_load_waitcnt + residual_load_waitcnt + fn_load_waitcnt*2)   // x 线程
vmcnt(            residual_load_waitcnt + fn_load_waitcnt*2)        // 非 x 线程
s_barrier
// 第二段:再收紧一档
vmcnt(x_load_waitcnt + residual_load_waitcnt + fn_load_waitcnt)
vmcnt(            residual_load_waitcnt + fn_load_waitcnt)
```

### `compute_store_tile()` —— 关键优化点

```cpp
// 1) 读 x / residual(自己那 4 个 k)
x_vec        = s_x[m_lane*tile_k + k_base]                  // 1× ds_read
residual_vec = s_residual[... + h*tile_mk]   for h in 0..3  // 4× ds_read

// 2) 【关键】把 24 个 fn 的 LDS 读一次性全发出去,再统一等 —— 不要读一个用一个!
for n in 0..24: fn_v[n] = s_fn[n*tile_k + k_base]           // 24× ds_read 背靠背发射
s_waitcnt_lgkmcnt(tile_n)   // 只等 x+residual(它们排在 fn 前面),fn 继续在途

// 3) 逐元素算 res + 累加 sqrsum_part + 写 next_residual
res[k] = x_vec[k]*post_mix_v + Σ_h residual_vec[h][k]*comb_mix[h]
sqrsum_part += Σ_k res[k]^2
store_vector(g_nres, res, ...)                              // 写 next_residual

// 4) GEMM:packed-FMA,2 个 k 一拍
s_waitcnt_lgkmcnt(0)        // 此时 fn 应已到齐
for n in 0..24:
    for kk in {0,2}:
        acc[n] = pk_fma_f32({fn_v[n][kk], fn_v[n][kk+1]},
                            {res[kk],     res[kk+1]}, acc[n])
```

`pk_fma_f32`(新增于 `csrc/include/aiter_opus_plus.h`,紧邻 `pk_mul_f32`):
```cpp
OPUS_D fp32x2_t pk_fma_f32(fp32x2_t a, fp32x2_t b, fp32x2_t c) {
    // gfx906/908/90a/94x/950 都有 v_pk_fma_f32;非 volatile,让调度器交错 24 个独立 FMA
    asm("v_pk_fma_f32 %0, %1, %2, %3" : "+v"(c) : "v"(a), "v"(b));
    return c;
}
```

---

## 5. 性能历程(MI355X, m=2048)

| 阶段 | kernel 时间 |
|---|---|
| 朴素串行 pipeline(每 iter `vmcnt(0)`+双 barrier) | 1047 us |
| + 2-stage 重叠预取(移植 mfma 流水) | 989 us |
| + fn LDS 读向量化 (b128) + 非 volatile `pk_fma` | 735 us |
| **+ 24 个 fn LDS 读一次性发射再统一等待**(最关键一步) | 163 us |
| + 拆分 `lgkmcnt` 等待(res 算术与 fn 读重叠) | **160–165 us(当前)** |
| —— 对照:原 mfma baseline | **153 us** |

> 关键洞察:735→163 us 这一跳来自 fn 的 LDS 读不再「读一个 FMA 一个」(每次都
> 卡在 ~20–30 cyc 的 LDS 延迟上,24 条串行依赖链),改为背靠背发射所有读、统一
> 一次 `lgkmcnt` 等待,把延迟重叠掉。

---

## 6. 诊断与瓶颈分析

用 `#ifdef MHC_FMA_SKIP_GEMM`(源码中保留)砍掉 GEMM 单独测访存地板:

- **GEMM 仅占 ~37 us**;FMA 计算本身不是瓶颈(印证 pk_fma == mfma 吞吐)。
- **访存地板 = 126 us**,而 ~263 MB HBM 只跑出 **2.1 TB/s**(MI355 元素级 kernel
  可达 6T+)→ kernel 是 **延迟 / 占用受限(1 wave/SIMD),不是带宽受限**。
- VGPR = 74,**0 spill**(`tile_m=8, tile_k=32`)→ 当前配置不受寄存器限制。

### 试过但都更差的方向(均已回退)

| 配置 | 结果 | 原因 |
|---|---|---|
| `tile_k=64`(换 16B 大 load) | 地板 126→**141 us** | 不是 load 粒度问题 |
| `tile_m=4`(grid 512,2 wave/SIMD) | 地板 126→**153 us** | 4B 细 load 损失盖过占用收益 |
| `tile_m=8 + tile_k=64`(满 CU+大 load) | **230 us** | `fn_v[24]×8 = 192 VGPR` 溢出 |
| 上一条 + `FN_G=4` 分组读 fn(限 VGPR) | **446 us** | 每组 `lgkmcnt(0)` 又把 GEMM 串行化 |

**核心矛盾:** 满 CU(`tile_m=8`)→ `tile_mk=256 < 512` → 退化成 4B load;要 16B
load 得 `tile_k=64` → `fn_v` 192 VGPR 溢出;而「一次性发射所有 fn 读 + 单次等待」
是 163 us 的命门,无法和「限制 VGPR」共存。

---

## 7. 结论与未尽事项

- **FMA 的前提(省带宽 617→340 MB)在此 kernel 不成立** —— 它是延迟/占用受限而
  非带宽受限,省下的带宽没有转化为加速。当前 FMA 版 160–165 us,仍比 mfma 的
  153 us 慢 ~10 us。
- **唯一还没堵死的路**:把 `fn` 在 LDS 里存成 bf16(VGPR 减半,或许让 `tile_k=64`
  不溢出,同时拿到「满 CU + 16B load + 单次等待」)。代价是 GEMM 前要做 bf16→fp32
  转换,精度需复核。

### 相关文件
- kernel:`aiter/csrc/kernels/mhc_kernels.cu` — `mhc_fused_post_pre_gemm_sqrsum_fma_kernel`
- helper:`csrc/include/aiter_opus_plus.h` — `pk_fma_f32`;`mhc_kernels.cu` 顶部 `cross_kg_sum`
- dispatch:`MHC_FUSED_POST_PRE_GEMM_SQRSUM_FMA_IMPL` 宏,在 `..._DISPATCH` 中当
  `hc_mult3==24 && split_k==1` 时走 FMA 路径,`grid=(ceil(m/8),1,split_k)`
- 测量:`python3 /tmp/measure_mhc.py 3`
- mfma 原版备份:`/tmp/mhc_kernels.cu.bak_premfma`
- 编译/测量流程:先 `AITER_REBUILD=1` 编译,再 `AITER_LOG_MORE=1` 读结果,
  **两者不可同时开**

---

## 附:完整 kernel 源码

```cpp
// csrc/include/aiter_opus_plus.h —— packed fp32 FMA helper
OPUS_D fp32x2_t pk_fma_f32(fp32x2_t a, fp32x2_t b, fp32x2_t c)
{
#if defined(__gfx906__) || defined(__gfx908__) || defined(__gfx90a__) || \
    defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__) || \
    defined(__gfx950__)
    asm("v_pk_fma_f32 %0, %1, %2, %3" : "+v"(c) : "v"(a), "v"(b));
    return c;
#else
    return fp32x2_t{a[0] * b[0] + c[0], a[1] * b[1] + c[1]};
#endif
}
```

```cpp
// mhc_kernels.cu —— 跨 k-group 蝶形归约
template <int tile_m>
__device__ float cross_kg_sum(float val, int lane_id) {
    static constexpr int warp_size = 64;
    #pragma unroll
    for(int s = tile_m; s < warp_size; s <<= 1) {
        int ival = __builtin_bit_cast(int, val);
        val += __builtin_bit_cast(float, __builtin_amdgcn_ds_bpermute((lane_id ^ s) * 4, ival));
    }
    return val;
}
```

完整 `mhc_fused_post_pre_gemm_sqrsum_fma_kernel` 见 `mhc_kernels.cu:1593-1843`
(过长不在此重复;上面第 3–4 节已逐段拆解)。
```
