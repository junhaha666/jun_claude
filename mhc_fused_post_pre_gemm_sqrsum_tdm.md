# `mhc_fused_post_pre_gemm_sqrsum_kernel` 的 TDM 改造

记录：把该 kernel 的 **residual load** 在 gfx1250 上从 `async_load` 改成 **单条 3D TDM tile**。
文件：`csrc/kernels/mhc_kernels.cu`。TDM 通用用法见 `gfx1250_opus_tdm_guide.md`。

> 范围限定：**只有 residual load 用 TDM**；`x`（layer_input）和 `fn` 保持原来的 `async_load` /
> VGPR 加载不变。所有改动都在 `#if defined(__gfx1250__)` 里，不影响 gfx942/gfx950 构建。
> **无法在 gfx950 机器上编译验证**，仅做修改。

---

## 1. 背景：这个 kernel 在干什么

融合 post + pre-gemm + sqrsum。每个 block 处理 `tile_m` 行、`tile_n` 列（hc_mult3 方向）、按
`tile_k` 沿 hidden 切 K：
- `s_x[n_stages * tile_m * tile_k]`：layer_input tile（async_load）。
- `s_residual[n_stages * tile_m * hc_mult * tile_k]`：residual tile（**本次改成 TDM**）。
- `fn` tile 直接进 VGPR。
- `compute_store_tile`：从 LDS 读 x + residual，算 `post_mix*x + Σ comb_mix*residual` →
  写 `next_residual`（global），并做 MFMA/WMMA 累加到 `v_cf`、累加 `sqrsum`。
- 双 buffer 预取（`n_stages=2`）。

gfx1250 在最新 main 上是 **wave32**：`warp_size=32`、`num_warps==hc_mult==4`、`block_size=128`。
`warp_id ∈ {0,1,2,3}` 正好对应 residual 的 4 个 head。

---

## 2. residual 的内存布局（关键，决定 TDM descriptor）

**Global**：`residual` 形状 `(m, hc_mult, hidden)`，`residual_stride = hc_mult*hidden = hc_hidden_size`。
元素 `residual[row][head][h]` 偏移 = `row*residual_stride + head*hidden_size + h`。
`residual_ptr = residual + idx*residual_stride`（idx = blockIdx.x * tile_m）。

**LDS（改造前后必须一致）**：`s_residual` 每 stage `hc_mult*tile_mk` 元素，按
`h*tile_mk + row*tile_k + hidden`（head 最外，head 内 row-major，行 pitch = tile_k，无 padding）。
`compute_store_tile` 就是这么读的：
```cpp
int s_offset = b*band_mk + (lane_id % mfma_m)*tile_k + (lane_id / mfma_m)*vec_tile;
residual_vec[h] = *(s_residual_rd_ptr + s_offset + h * tile_mk);   // band_mk = mfma_m*tile_k
```
**目标：TDM 产出的 LDS 布局要精确等于这个**，这样 `compute_store_tile` 一行都不用改。

---

## 3. 为什么是"单条 3D tile"

TDM 的 LDS 写入顺序是 dim0→dim1→dim2（inner→outer），线性下标
`dim2*(td1*td0) + dim1*td0 + dim0`。要得到 `h*tile_mk + row*tile_k + hidden`，令：

| TDM 维 | 含义 | TileDim（静态） | tensor_dim（OOB 边界） | stride（元素） |
|---|---|---|---|---|
| dim0 | hidden | `tile_k` | `tile_k` | 连续（隐含 1） |
| dim1 | row | `tile_m` | `m_oob` | `residual_stride`（=s0） |
| dim2 | head | `hc_mult` | `hc_mult` | `hidden_size`（=s1） |

于是 LDS 线性 = `head*(tile_m*tile_k) + row*tile_k + hidden` = `h*tile_mk + row*tile_k + hidden` ✓。

**为什么不是别的方案**（inflight ≤ 3 的硬约束）：
- 按 head 拆 4 条 2D tile → 单 stage 就 4 条 inflight，违反 ≤3。
- 2D 合并 (row,head) 成一维 → LDS 变 `[row][head][hidden]`，得改 `compute_store_tile` 读偏移。
- **单条 3D**：每 stage 1 条，双 buffer = 2 条 inflight，且 compute 完全不改 → 选它。

---

## 4. 实际改动

### 4.1 文件顶部：TENSORcnt 宏（在 `mhc_async_load_oob_guard` 旁）
```cpp
#if defined(__gfx1250__)
#define MHC_TDM_KEEP1() opus::s_wait_tensorcnt(opus::number<1>{})   // 留下一条预取在飞
#define MHC_TDM_DRAIN() opus::s_wait_tensorcnt(opus::number<0>{})   // 全部 drain
#else
#define MHC_TDM_KEEP1() ((void)0)
#define MHC_TDM_DRAIN() ((void)0)
#endif
```

### 4.2 residual load lambda 拆 arch 分支
```cpp
#if defined(__gfx1250__)
    static constexpr int residual_load_waitcnt = 0;   // residual 归 TENSORcnt，不进 async 计数
    auto lds_load_residual_tile = [&](int k){
        if (warp_id == 0) {                            // wave 级 DMA，只让 warp0 发起整块（含全部 head）
            opus::tdm_window<DTYPE_I, tile_k, tile_m, hc_mult> win;   // ndim=3, cpol=default_cpol=16
            uint32_t lds = static_cast<uint32_t>(reinterpret_cast<__UINTPTR_TYPE__>(s_residual))
                + static_cast<uint32_t>((k & 1) * (hc_mult * tile_mk) * sizeof(DTYPE_I));
            DTYPE_I* g = residual_ptr + k_split_offset + k * tile_k;
            // make(lds, gptr, td0, td1, s0, s1, lds_bar, td2)
            win.desc.make(lds, g, tile_k, m_oob, residual_stride, hidden_size, 0, hc_mult);
            win.load_to_lds();
        }
    };
#else
    /* 原 async_load + oob_guard 版本，原样保留 */
#endif
```
- `tdm_window::make` 只有 2D → **手搓 `win.desc.make(...)`**（3D），再用官方 `win.load_to_lds()`
  发起（自动处理 5/6-operand + cpol）。见通用指南 §8。
- `tensor_dim1 = m_oob` → 越界行自动 load 0，替代原 oob_guard 的手动置零。
- `residual_load_waitcnt = 0`：原来它进 `x_async_wait/r_async_wait`；gfx1250 上 `mhc_async_load_oob_guard`
  为 true，那两个值本就被三元取 0，所以设 0 一致、不影响 x 的 async 等待。

### 4.3 同步接线
`wait_load_cnt` 开头加 `MHC_TDM_KEEP1();`（在 barrier 前，保证当前 residual stage 就位后再由
barrier 对全 block 发布）；三处尾部 drain（`s_wait_all_loadcnt(0_I, 0_I);` 之后、barrier 之前）
各加 `MHC_TDM_DRAIN();`。

```
wait_load_cnt():                          尾部（每个 compute 前的 drain 点）:
    MHC_TDM_KEEP1();          // 新增        s_wait_all_loadcnt(0_I, 0_I);
    s_wait_all_loadcnt(...);                 MHC_TDM_DRAIN();        // 新增
    s_barrier();                             s_barrier();
    s_wait_all_loadcnt(...);                 compute_store_tile(...);
```
TENSORcnt 与新的 `s_wait_all_loadcnt`（load/async 分离计数）互相独立，互不干扰。

---

## 5. inflight 时序（≤3 验证）

- prologue：`lds_load_residual_tile(0)`、`(1)` → warp0 发 2 条 TDM，2 条 inflight。
- 主循环每次 `wait_load_cnt` 里 `s_wait_tensorcnt(1)`：当前 stage 完成，留下一条预取；
  compute 后 barrier → prefetch 下一 stage（写另一个 slot，(k&1) 双 buffer 无覆写冲突）。
- 尾部 `s_wait_tensorcnt(0)` drain 干净。
- 全程 inflight ≤ 2 ≤ 3 ✓。

---

## 6. 未验证的假设（gfx950 不能编 gfx1250）

1. **3D tile 的 LDS 写序** = dim0→dim1→dim2（据 2D 参考推断；仓库
   `test_tdm_gfx1250.cu` 只有 2D 例子，无 3D 可对照）。
2. 用 `tdm_window` 但绕过它的 2D `make()`、直接填 `.desc` 再 `load_to_lds()` —— 合法但非标准；
   因此**不能再对该 win 调 `move()`**（状态没维护），每 k-step 重新 `desc.make()`。
3. `tensor_dim1=m_oob` 的行 OOB 行为与原 oob_guard 手动置零等价（TDM OOB load 返 0）。

---

## 7. 相关文件
- `csrc/kernels/mhc_kernels.cu`：本 kernel（`mhc_fused_post_pre_gemm_sqrsum_kernel`）。
- `gfx1250_opus_tdm_guide.md`：opus TDM 通用用法。
- `csrc/include/opus/opus.hpp`、`csrc/include/aiter_opus_plus.h`：TDM / 计数器 API。
