# MHC GEMM: fp32 MFMA → bf16 MFMA 优化记录

## 背景与动机

目标硬件:**AMD Instinct MI355X (gfx950)**。

`csrc/kernels/mhc_kernels.cu` 里 MHC 的两个 GEMM kernel 原本用 `__builtin_amdgcn_mfma_f32_16x16x4f32`
(fp32 输入、fp32 累加)。在 gfx950 上这条 **fp32 MFMA 由 VALU 执行**(不是 XDL/tensor 单元),
因此:

1. 占用 VALU 流水,无法与其它 VALU 指令并行(co-exec)。
2. K=4 很小,需要大量指令覆盖完整 K 维。

**核心诉求**:把 GEMM 从 VALU 卸载到 XDL,腾出 VALU,并减少 MFMA 指令数。

方案:换成 gfx950 原生 bf16 MFMA(`mfma_f32_16x16x32_bf16`,K=32,XDL 执行)。为保精度,
对 fp32 权重 `fn` 做 **hi/lo bf16 拆分**:
- `fn_hi = RNE-bf16(fn)`
- `fn_lo = bf16(fn - fp32(fn_hi))`,残差量级 ~2⁻⁸·|fn|
- `out = x·fn_hi + x·fn_lo`,恢复 fn 到 ~2⁻¹⁶ 尾数精度

**关键点**:bf16 与 fp32 共享 8 位指数,残差不会下溢,所以**无需 scale**(不像 fp16 那样需要)。

---

## 涉及的两个 kernel

| kernel | 行号 | 走的场景 | GEMM 操作数 |
|---|---|---|---|
| `mhc_fused_post_pre_gemm_sqrsum_kernel` | ~1643 | m ≤ 1024(真融合 post+pre+gemm) | 激活 res = 当场算的 **fp32** 中间量;权重 fn = fp32 |
| `mhc_pre_gemm_sqrsum_kernel` | ~37 | m ≥ 2048(large_m 路径:mhc_post + mhc_pre) | 激活 x = 天然 **bf16**;权重 fn = fp32 |

### 为什么大 m 不走融合 kernel

`mhc_fused_post_pre`(`aiter/ops/mhc.py`)在 gfx950 上 m>1024 走 `large_m` 分支 = 拆成独立的
`mhc_post` + `mhc_pre`(注释:大 m 时拆开更快,因融合 kernel `launch_bounds(...,1)` + 重 LDS,
occupancy 受限)。GEMM 因此落在 `mhc_pre_gemm_sqrsum_kernel`,且 `mhc_pre` 用 split-k 切 K。

### 两个 kernel 的归约差异(重要)

- **pre_gemm**:warp 间按 tile_m(m 行)切分,完整 K 维在单 warp 内闭合 → 每 warp 结果即最终,
  **无需跨 warp 归约**。store 前本地 `v_cf + v_cf_lo/scale` 即可。
- **融合版**:`block_size == hc_mult × warp_size`,每 warp 负责一个 hc head,K 维的 hc_mult 段被切到
  不同 warp,多 warp 对同一输出算部分积 → **必须跨 warp LDS 归约**(代码 ~1948)。
  scale 版采用「每 warp 先本地合并 `v_cf += v_cf_lo/scale`,再走原归约」——因合并与归约都是线性,
  等价于先归约后合并,但归约代码完全不动。

---

## 关键实现细节

### K 维布局(决定 bf16x8 打包可行性)

pre_gemm 的 x 加载(`load_vector_nbytes<..., vec_tile, 8*sizeof(T), interleave=4>`):
- `v_a[i*8+e]` → K_global = i*32 + g*8 + e,g = lane_id/mfma_m(lane-group 0..3)

fp32 MFMA 的 K=4 来自 4 个 lane-group;fn 的 `K_wanted = (kk/8*4 + g)*8 + kk%8 = i*32+g*8+e`,
与 x **完全一致**。一个 chunk(8 个连续 kk × 4 lane-group)= 32 个 K,正好匹配 `16x16x32_bf16`
的布局。**LDS swizzle 布局完全不用改**。

### 转换时机:延迟到 MFMA 前(co-exec 关键)

fn 的 fp32→bf16 hi/lo 拆分**不在 load 时做**,而是放到 MFMA 循环里现做。这样:
- fn 仍以 fp32 走 load + 双缓冲,不破坏 2-buffer prefetch。
- 转换是 VALU、MFMA 是 XDL,二者 **co-exec**;转换指令藏在 MFMA 后面。

实测(融合版 m=1024):延迟转换比 load 时转换在 m≥128 普遍快 1.5~5μs。

### 用到的 opus 工具

- `opus::mfma_f32_16x16x32_bf16{}(a, b, c)` — gfx950 原生(opus.hpp:2264/2312)
- `opus::fp32_to_bf16(float)` — opus.hpp:1109(gfx950 单条硬件指令)
- `opus::bf16_to_fp32(bf16_t)` — opus.hpp:1099(`<<16`,近乎免费)
- 泛型:pre_gemm 用 `opus::mfma<DTYPE_I,DTYPE_I,fp32,16,16,32>` + `static_cast<DTYPE_I>` 兼容 fp16/bf16

---

## 性能与精度结果(n=7168, hc_mult=4, fuse_rmsnorm)

测试:`python3 op_tests/test_mhc.py -n 7168 --fuse_rmsnorm`(跑两次,取第二次稳定值)。
精度容差 atol/rtol = 0.01。

### A. mhc_pre_gemm_sqrsum_kernel(mhc_pre summary 的 `hip_us`,大 m 走这条)

| m | 原版 fp32 | bf16 无scale | bf16 +scale=256 |
|---:|---:|---:|---:|
| 256 | 18.87 | **17.46** | 17.85 |
| 512 | 25.57 | **22.95** | 24.09 |
| 1024 | 39.89 | **34.60** | 36.67 |
| 2048 | 69.19 | **60.43** | 61.46 |
| 8192 | 251.76 | **224.28** | 228.03 |
| 65536 | 1911.65 | **1693.27** | 1721.86 |

bf16 无scale vs 原版:**大 m +11~13%**,中等 m +7~13%。
精度 `hip_err` 全 **0**。

### B. 融合版(mhc_post_pre summary 的 `fused_us`,m≤1024)

| m | 原版 fp32 | bf16 无scale | bf16 +scale=256 |
|---:|---:|---:|---:|
| 256 | 24.00 | **22.46** | 22.76 |
| 512 | 37.85 | **34.28** | 34.78 |
| 1024 | 64.79 | **59.54** | 60.44 |

bf16 vs 原版:**+5~9%**(融合版 m≤1024 访存受限,收益小于 pre_gemm)。
精度 `fused_err`:原版 1e-2~1e-3 → bf16 **全 0**(精度反而更好,因 ref 的激活本就是 bf16)。

### C. 大 m 端到端(mhc_post_pre summary 的 `unfused_us`)

| m | baseline fp32 | bf16 |
|---:|---:|---:|
| 2048 | 111.65 | **102.82** (+7.9%) |
| 8192 | 438.83 | **417.76** (+4.8%) |
| 65536 | 3418.08 | **3226.79** (+5.6%) |

---

## rocprof 验证(融合版 m=1024,bf16 无scale vs 原版 fp32)

| 计数器 | 原版 fp32 | bf16 延迟转换 | 变化 |
|---|---:|---:|---:|
| SQ_INSTS_MFMA | 917,504 | 229,376 | **−75%**(÷4) |
| SQ_INSTS_VALU | 6,070,784 | 6,517,248 | +7.4% |
| SQ_BUSY_CYCLES | 3,081,023 | 2,788,414 | **−9.5%** |
| **SQ_VALU_MFMA_COEXEC_CYCLES** | **0** | **625,734** | **0 → 有** |

pre_gemm(m=8192,改造前 baseline):MFMA=7.34M, VALU=17.83M, COEXEC=0
→ 印证大 m 是算力受限(MFMA 量巨大),bf16 收益更大。

**结论被数据证实**:MFMA 指令降到 1/4、GEMM 从 VALU 卸到 XDL、转换 VALU 与 MFMA 真正 co-exec。

---

## scale=256 已移除(两个 kernel)

2026-06-16:**pre_gemm 与融合版都已去掉 scale**,改单累加器(`v_cf`,hi*x 与 lo*x 直接累加)。
融合版同样删掉了 `v_cf_lo` 和跨 warp 归约前的合并循环;精度全程 err≈1e-8(=0)。
融合版 fused_us:256 −1.0%、1024 −1.3%(中小 m VGPR 压力敏感,去 `v_cf_lo` 受益)。

### 为什么 bf16 下 fn_lo 不需要乘 scale=256

直觉误区:"fn_lo 是个很小的残差(≈fn 的 1/256),直接转 bf16 会损失很多精度,得先乘 256 放大"。
这个直觉对**定点数**成立,但 **bf16 是浮点数**,所以不成立。

**根因:bf16 与 fp32 共享同样的 8 位指数。**
```
fp32:  [sign 1][exp 8][mantissa 23]
bf16:  [sign 1][exp 8][mantissa 7]   ← 指数位宽完全一样
```
bf16 就是 fp32 砍掉低 16 位 mantissa。两者**数值范围相同**,只是有效位数从 23→7。
浮点数的**相对精度恒定 = 2⁻⁷**,与这个数本身的量级无关。

**hi/lo 拆分在做什么:**
```
fn_hi    = bf16(fn)              // 取 fn 高 7 位 mantissa
residual = fn - fp32(fn_hi)      // 量级 ≈ 2⁻⁸·|fn|,但它是个独立浮点数,自带指数
fn_lo    = bf16(residual)        // 取 residual 的高 7 位 mantissa
```
关键:`residual` 虽然比 `fn` 小 ~256 倍,但它**自带自己的指数**(比 fn 指数小约 8)。
因为 bf16 指数范围和 fp32 一样宽,这个小残差塞进 bf16 时**指数完全不丢**,只是 mantissa 取
自己的高 7 位。于是 hi(7位)+ lo(7位)≈ **14~16 位有效 mantissa**,误差 ~2⁻¹⁶,远在 0.01 容差内。

**数值例子:**
```
fn       = 1.1009
fn_hi    = bf16(1.1009)    ≈ 1.1015625      // 误差 6.6e-4
residual = 1.1009 - 1.1015625 = -6.625e-4   // 独立浮点数,指数 ~2⁻¹¹
fn_lo    = bf16(-6.625e-4) ≈ -6.6185e-4     // 7位 mantissa,误差仅 ~5e-6
hi + lo  = 1.10090065       vs 1.1009        // 最终误差 ~6e-7
```

**scale=256 到底改了什么、为什么无效:**
```
residual*256 = -0.16965
bf16(-0.16965) 误差 = 0.16965·2⁻⁷ ≈ 1.3e-3
再 /256 → 5.2e-6   ← 与不乘 scale 的 5e-6 完全一样
```
乘 256 只是把 residual 的**指数**抬高 8,而 bf16 的 mantissa 位数(7)**不随指数变**。
浮点相对精度只看 mantissa 位数,所以 scale 对 bf16 是**纯 no-op**,白白多一条 `v_pk_mul`
(trace ~186K latency)+ 一组累加器。

**scale 只对 fp16 有意义:** fp16 = `[sign 1][exp 5][mantissa 10]`,**指数仅 5 位**,范围窄
(最小正规数 ~2⁻¹⁴)。若 fn 本身小、residual 再小 256 倍,可能**下溢到 denormal 或 0**(指数位不够)。
此时乘 256 把 residual 抬回 fp16 正规数区间才保住 mantissa。bf16 指数宽,不会下溢,故无需 scale。

### pre_gemm 细节
- **精度**:全程 err ≈ 1e-8(=0),证实 bf16 下 scale 零收益(bf16 与 fp32 同 8 位指数,
  残差 `fn - fp32(hi)` 自带指数,转 bf16 不下溢,mantissa 相对精度恒定 2⁻⁷)。
- **性能**:unfused_us m=65536 3264→3107(−4.8%),中等 m 持平。
- **VGPR**:去掉 `v_cf_lo` 后 arch_vgpr=48(occupancy 有余量)。
- 消掉了:`v_pk_mul`(*256,trace ~186K latency)、半组累加器、store 前合并循环。

rocprof(m=8192,去 scale 后,per-dispatch):VALU=985,728 / MFMA=57,344 / COEXEC=251,317。

### 剩余瓶颈(待解)

trace 里 VALU 仍是 MFMA 的 ~17 倍。去 scale 只砍了 ~19% 的转换 VALU。剩下 ~780K 全来自 **lo 路径**
(`v_cvt`(lo 半)、`v_lshlrev`+`v_and`(`bf16_to_fp32(hi)`)、`v_pk_add`(`fn-fp32(hi)` 减法))。
唯一能砍掉的途径是 **hi-only**(放弃 lo),精度 ~2⁻⁸≈0.4%,需实测 K=28672 累加后能否过 0.01 容差。

## 融合版 load 调度优化(2026-06-16,m=4096,splitK off)

ATT trace(`rocprofv3 -i inputs.json`,kernel_include=`mhc_fused_post_pre_gemm_sqrsum_kernel`)
显示融合版瓶颈是**访存 stall**(关 splitK 后更重),前三项占总 stall 82%:
`s_waitcnt`(328K)+ `buffer_load_dwordx4`(232K)+ `s_barrier`(209K)。

根因(两个):
1. **fn 的同步 buffer_load 排在 s_barrier 之后**,延迟(单条最高 33984)没被 compute 掩盖。
   fn 走 VGPR(`vgpr_load_fn_tile`),不像 x/residual 走 async_load+LDS 双缓冲。
2. **一次 fn load 是 4 条连续 dwordx4**(repeat_n=2 × 每 n 2 条 16B),背靠背 issue 堵死 memory
   issue port(单处 4 连发累计 stall 28184)。奇偶迭代排布不对称,只有一路出现 4 连发。

改法:
1. **fn prefetch 提到 s_barrier 之前**:fn 不碰 LDS,与 barrier 无关,提前发让其延迟与 barrier
   等待 + LDS prefetch 重叠。→ buffer_load stall −55%。
2. **打散 fn load**:新增 `vgpr_load_fn_n`(每次只发一个 n),在主循环里把 n=0 / n=1 两次 load
   用 LDS prefetch 隔开 → 4 连发拆成 2+2,消除 issue port 堵塞(>=3 连发 run 数 4→0)。
   → s_barrier stall −30%、s_waitcnt −22%。

结果(stall 累计):936,936 → 873,812 → **815,964(−12.9%)**。
端到端 fused_us(干净测时):241 → **229~234(−3~5%)**。精度全程 err≈0。

源码:`vgpr_load_fn_n` lambda + 主循环/epilogue 重排(fn prefetch 前移 + per-n 打散 + sched_barrier 隔离)。

### 剩余(融合版)
stall 仍以 s_waitcnt(293K)为主——load 延迟本身仍在,只是不再自堵。进一步要么加深 prefetch
(3-buffer,VGPR 受限),要么减少 fn load 量(hi-only 同样能砍 fn 转换,但 load 量不变)。

## (历史)scale=256 的结论(已尝试,不建议保留)

需求方曾要求把 `fn_lo` 乘 256、相加时再除 256(双累加器 `v_cf` + `v_cf_lo`)。结果:

- **精度无收益**:无 scale 的 RNE 版已是 0 误差(bf16 与 fp32 同指数,残差不下溢)。
- **性能略降**:
  - pre_gemm:大 m −1.7%,中等 m(512~1024)−5%(VGPR 压力在 occupancy 受限段更敏感)。
  - 融合版:−1.5%。
- 多一组累加器(`v_cf_lo`)增加 VGPR 占用,中小 m 段性能损失更明显。

**scale 仅在 fp16 路径(5 位指数,残差易下溢)才有意义。bf16 下不必要。**
当前源码是 scale 版;若以 bf16 为主,建议回退到无 scale 版以拿回 1.5~5% 性能。

---

## 当前源码状态

`csrc/kernels/mhc_kernels.cu`:两个 kernel 均为 **bf16 MFMA + RNE hi/lo + 单累加器(无 scale)**。
- pre_gemm:单 `v_cf`,hi*x 与 lo*x 直接累加,store 前无合并。
- 融合版:单 `v_cf`,hi*x 与 lo*x 直接累加,跨 warp 归约前无合并。
- scale=256 已于 2026-06-16 移除(详见上文「为什么 bf16 下 fn_lo 不需要乘 scale」)。

待定:hi-only(放弃 lo)能否过 0.01 容差——能过则 VALU −80%、MFMA −50%。
