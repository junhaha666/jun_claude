# MHC kernel 优化工程记录

目标 GPU：MI355X（gfx950，256 CU，~8TB/s HBM）。涉及文件：
`csrc/kernels/mhc_kernels.cu`、`aiter/ops/mhc.py`、测试 `op_tests/test_mhc.py`。

主要优化对象：小 m（m≤256）下 `mhc_pre` / `mhc_fused_post_pre` 两条路径里的
latency-bound kernel。基准用例 `m=32, hidden=7168`。

---

## 0. 构建 / 测量 / 抓 trace

- 重新编译：`AITER_REBUILD=1 python3 op_tests/test_mhc.py -n 7168 --fuse_rmsnorm -m 32`
  （~35s/module）。
- 单 kernel device time：`AITER_LOG_MORE=1 python3 op_tests/test_mhc.py -n 7168 --fuse_rmsnorm -m 32`。
- **绝不能同时开 `AITER_REBUILD=1` 和 `AITER_LOG_MORE=1`**。
- ATT trace：`rocprofv3 -i inputs.json -d ./att-logs -- python3 op_tests/test_mhc.py -n 7168 --fuse_rmsnorm -m 256`
  （`inputs.json` 的 `kernel_include_regex` 指定目标 kernel）。
- 指定 GPU tune：`export HIP_VISIBLE_DEVICES=3`（用 rank3）。
- 测试用**未播种 randn**，每次数据不同 → 偶发 `warning!` 百分比逐次变化是数据差异，
  不一定是 race。
- **device-time 噪声带 ~0.5us（小 m）/ probe 单次方差 ±3-5%**。比较两版必须多轮取
  min/median，单次数值不可信。

---

## 1. 诊断：小 m 是 launch / first-load-latency bound

- m=32 时 `mhc_pre_big_fuse_rmsnorm_kernel` 早期基线 11.2us，0.14 TB/s（~2% 峰值带宽）。
  num_rows=2 只发 16 个 block 到 256 CU → 94% GPU 空闲。
- 改 num_rows=1（32 block）几乎不帮 → **不是 occupancy-bound**。
- sinkhorn_repeat 20→0 无变化；split-k 112→1 只省 ~0.2us。说明 A/B 顶部归约的成本
  **不是循环迭代数**，而是 **dependent global-load latency**（首个 HBM load 暴露在
  block 开头，没有别的工作可以掩盖）。
- 结论：小 m 下 kernel 受 **launch / first-load latency** 支配，GPU 90%+ 空闲。
  传统旋钮（split-k / num_rows / sinkhorn / fast-div）边际收益已耗尽，真正的杠杆是
  **结构性的**：减少跨 kernel 的 HBM 往返、缩短 per-block 串行延迟链。

详细 trace 分解见 memory `project_mhc_rmsnorm_perf.md`。

---

## 2. 已落地的优化

### (a) num_rows=1, residual_block=1024（小 m，rmsnorm kernel）

`MHC_PRE_BIG_FUSE_RM_KERNEL_DISPATCH` 按 m 分档：小 m 用 nr=1/block=1024，大 m 保留
nr=2/512 调优路径。结果 m=32：kernel 11.2→8.8us，total 16.7→13.3us（~20%）。

- **附带修了一个潜伏 UB bug**：comb-warp 分支里 `comb_mix_v`/`post_mix_v` 未初始化，
  只对 `lane < num_rows*hc_mult2` 赋值，但 DPP 跨 lane 归约会读全部 64 lane 的寄存器。
  非活跃 lane 的未初始化 VGPR = C++ UB，printf 会掩盖（heisenbug）。修复：初始化为
  0.0f。nr=2 时也潜伏（碰巧更多 lane 活跃）。

### (b) defer-consume：重叠 A/B 两条顶部 load 流（rmsnorm kernel）

A（sqrsum→rms）和 B（gemm_out_mul）原本被中间 `__syncthreads` 串行，暴露两次独立
first-load latency。关键教训：**要重叠两条 load 流，必须 defer 的是 CONSUME（触发
s_waitcnt 的那个 `+=`），不是仅仅挪动 load 指令位置。**

- defer-A：sqrsum load 进寄存器 `v_sq` 先不消费，跑完 B 的 load+accumulate 再消费。
  ATT trace（m=256）：total lat 66496→58480（-12%）。
- defer-B：B 的首批 gemm load 也 hold 进寄存器，先做 rms reduce（与 gemm 无关），再消费。
  trace 累计 66496→56564（-15%）。
- **注意**：trace lat 改善明显，但 wall-clock 落在噪声带内（GPU 空闲吃掉了 stall 改善）。
- 同样的 defer-A 也加到了非 rmsnorm 的 `mhc_pre_big_fuse_kernel`（行为一致，正确性通过）。

### (c) ⭐ in-kernel hc_mult 归约（fused 路径，最实在的结构性win）

**问题**：同一个 rmsnorm kernel 在 `mhc_pre` 里 7.1us、在 `mhc_fused_post_pre` 里 8.2us。
根因是两条路径喂的 n_splits 不同：
- `mhc_pre`：n_splits = `selected_splitk`（112）
- `mhc_fused_post_pre`：n_splits = `selected_splitk * hc_mult`（448）

后者 4 倍，因为 producer `mhc_fused_post_pre_gemm_sqrsum_kernel` 用 block_size=hc_mult*64，
4 个 warp（warp_id=head）各写自己 head 的 partial。consumer 的 latency-bound A/B 归约
要从 HBM 多读 4 倍 partial。

**修复**：在 producer 内用 LDS 跨 4 warp 归约 `v_cf`（gemm_out_mul）和 `sqrsum_part`，
只写 split_k 份 partial。hc_mult sum 本就是 GEMM K 维收缩（sum over hc_mult*hidden）的
一部分，fp32 LDS 相加 → 数学精确。固定 lane_id 时 4 个 warp 持有的是**同一输出元素**的
贡献。`next_residual`（per-head 输出）不变。

涉及三处改动：
1. kernel：k_loop 后加跨 warp 归约（复用 `s_residual` 作 float scratch，loop 后已死），
   warp 0 写 `k_split_idx*m`；`out_ptr`/`sqrsum` 索引去掉 `warp_id`。
2. host wrapper：`split_k = gemm_out_sqrsum.size(0)`（原 `/hc_mult`），TORCH_CHECK 改 `==split_k`。
3. python `mhc.py`：`n_splits = selected_splitk`（去掉 `*hc_mult`）。consumer 通过
   `gemm_out_mul.size(0)` 自动拿到 112。

**结果**（m=32, n=7168）：fused 路径 rmsnorm kernel **8.2→7.0-7.6us**；producer
6.3→5.9-6.3us（无回退）；净收益正，另省 ~1MB HBM 写+读。正确性通过（post_mix/
layer_input 的 warning 是已知 rcp 精度，unfused baseline 也有）。

---

## 3. 重新 tune fused producer（2026-06-08，HIP_VISIBLE_DEVICES=3）

在 (c) 改动后，按 `mhc_fused_post_pre_tuning.md` 重新 tune `_mhc_fused_config_gfx950_256`，
sweep hidden=4096/7168、m=1…16384 所有 2 的幂。

**探针修正**：`/tmp/probe_mhc.py` 的 `make_inputs` 必须用 `n = splitk`（原 `splitk*hc_mult`），
因为 producer 现在 in-kernel 归约后只输出 split_k 份。

**结论：现有 policy 无需改动**。in-kernel hc_mult 归约**没有移动最优配置**（producer
每 block 多的那次跨 warp LDS 归约不改变工作画像）。对比每个 m 实测最优：
- m≤256：launch-bound ~7.5us 平台，config 无所谓，3-4% gap 是噪声。
- m≥512：policy 命中 0-2%。
- 唯一离群 m=128/h=7168 4%（0.4us，低于「<2-3% 不编码」阈值）。

tk crossover（`m≥2*hidden`→tk=64）和 split_k 几何填充仍正确落位。

---

## 4. 尝试过但无可靠收益（A/B 实测）

### bank-conflict 修复 + 合并 barrier（in-kernel 归约 tail）

(c) 的归约 tail 有两个理论问题：
1. LDS 布局 `[warp][lane][c]`，lane 维 stride = v_per_lane（small-m 时 8 floats）→
   8-way bank conflict。改成 `[warp][c][lane]`（lane 连续）可消除。
2. sqrsum 用了独立的 deposit→sync→accumulate，多 2 个 barrier。可与 v_cf 共用同一对 barrier。

实现后 A/B（min over 8 rounds，build 在同会话）：

| m | 基线(bank-conflict, 4 barrier) | 新版(bank-fix, 2 barrier) | 差异 |
|---|---|---|---|
| 32 | 8.40us | 8.65us | +3% 慢 |
| 256 | 15.52us | 15.34us | -1% |
| 1024 | 42.31us | 43.15us | +2% 慢 |
| 4096 | 157.14us | 150.01us | -5% 快 |

**结论：无一致信号，全在方差带内（probe 单次方差 ±3-5%）。** 归约 tail 只是 kernel 尾巴，
K-loop（读 hidden=7168）才是大头，bank-conflict 改善的 LDS 延迟被整体淹没。理论上更干净
（无 bank conflict、少 2 barrier、正确性通过），但不产生可测 wall-clock 收益。

新版保存在 `/tmp/mhc_new.cu`。**当前 tree 仍是基线版**；保留/回退取舍未定（边际，等定夺）。

### 其它已排除

- **atomic-add 替代 s_hc_mult3_partial**：~10% 更慢（53-way 串行 LDS atomic 竞争 + 额外
  barrier）。回退。
- **强制 split-k=1 快路径**：split-k 自然只在 m≥32768 才到 1；强制会让上游 gemm_sqrsum
  慢 30x。回退。
- **A/B reorder（不 defer consume）**：无效（consume 仍触发 waitcnt，两流仍串行）。
- **多累加器 stage-2 reduce（RED_ACC=4）**：无效但无害。
- **defer-B 在 wall-clock**：trace 改善但 device-time 反而 7.7us（噪声带），回退过一版。

---

## 5. 已知非 bug

- **comb_mix 的 rcp 精度 warning**：delta ~0.04-0.05，2-4% 元素。是 sinkhorn 用 rcp
  （fast reciprocal）指令导致，**不是 bug，忽略**。unfused / triton baseline 同样有。
- in-kernel hc_mult 归约是 fp32 精确，任何真正的归约 bug 会是大的、结构化的误差，
  不会是散点近阈值 delta。

---

## 6. 剩余可探索方向（结构性，未做）

- 把 `gemm_sqrsum` kernel 与 consumer kernel 进一步融合，避免第二次 kernel launch +
  从 HBM 重读 gemm_out（gemm 输出就是 A/B 的输入）。
- C 阶段 residual load→LDS（~30% stall）用同样的 defer-consume 技巧。
- prologue first-load latency（~38%）在小 m 下基本不可约（block 开头无前置工作可掩盖）。
