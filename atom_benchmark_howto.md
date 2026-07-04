# ATOM Benchmark 复现方法 (DeepSeek-V4-Pro)

复现 CI "ATOM Benchmark" workflow 里某个 cell 的性能数据。以
`DeepSeek-V4-Pro (isl=1024 osl=1024 c=32)` 为例。

CI 来源：`.github/workflows/atom-benchmark.yaml`，实际跑两步
`atom_test.sh launch` + `atom_test.sh benchmark`。本机已经在容器内
(gfx950 8 卡, `/app/ATOM`)，所以直接调脚本，不走 docker。

---

## 0. 环境与路径

| 项 | 值 |
|---|---|
| ATOM 仓库 | `/app/ATOM` |
| aiter 仓库 | `/app/aiter-test` |
| 模型 | `/shared/data/amd_int/models/deepseek-ai/DeepSeek-V4-Pro` |
| GPU | gfx950 ×8 (MI355 系) |
| server 端口 | 8000 |
| server 日志 | `/tmp/atom_server.log` |
| client 日志 | `/tmp/atom_client.log` |

CI 展开出来的这个 cell 的参数（用 `.github/scripts/build_benchmark_matrix.py` 生成）：

- `server_args`: `--kv_cache_dtype fp8 -tp 8 --hf-overrides '{"use_index_cache": true, "index_topk_freq": 4}'`
- `env_vars`: `AITER_BF16_FP8_MOE_BOUND=0`、`ATOM_MOE_GU_ITLV=1`
- `ISL=1024 OSL=1024 CONC=32 RANDOM_RANGE_RATIO=0.8`
- num-prompts = CONC×10 = 320，num-warmups = CONC×2 = 64

---

## 1. 必做的环境变量

### NO_PROXY（关键！否则全部请求失败）

机器上设了 `HTTP_PROXY` / `HTTPS_PROXY=http://66.42.115.190:18129`。
benchmark 客户端用 `aiohttp(trust_env=True)`，会把发往 `localhost:8000`
的请求全导去外部代理 → 320 个请求全 fail、server 一条都收不到、
6 秒"跑完"且 `Successful requests: 0`。

必须设：

```bash
export NO_PROXY=localhost,127.0.0.1
export no_proxy=localhost,127.0.0.1
```

CI 环境没有 proxy 变量，所以 CI 不受影响；本机复现一定要加。

### cell 业务环境变量

```bash
export AITER_BF16_FP8_MOE_BOUND=0
export ATOM_MOE_GU_ITLV=1
```

---

## 2. 每次测试前清 cache

```bash
rm -rf ~/.cache/atom ~/.triton/cache
```

- `~/.cache/atom`：ATOM 引擎层编译缓存（含运行时按 shape 触发的
  `module_pa_sparse_prefill_opus` 等稀疏 prefill kernel）。
- `~/.triton/cache`：triton kernel 缓存。

清 cache 的动作要在 **启动 server 之前** 做。
（注：`atom_test.sh launch` 内部本身也有 `rm -rf ~/.cache/atom/*`，
但为求干净，显式整目录清掉。）

清完后第一个真实请求会触发一次 JIT 编译（warmup 第 1 个请求会卡十几秒、
日志出现 `No available shared memory broadcast block`，属正常），编译完即热。

---

## 3. 切换版本时的额外步骤

### ATOM 换 commit

```bash
cd /app/ATOM && git checkout <commit>
```
ATOM 是 Python，切完直接清 cache 重启即可（无需编译 so）。

### aiter 换 commit（涉及 C++ 改动要删 so 触发重编译）

```bash
cd /app/aiter-test && git checkout <commit>
```

如果该 commit 区间改了 `csrc/` 或 `hsa/` 下的文件，必须删掉对应的
`aiter/jit/module_*.so`，否则用的还是旧编译产物。

**找出要删哪些 so**：`aiter/jit/optCompilerConfig.json` 是
module→源文件映射表。把区间内改动的 csrc 文件名（及被改 header 波及的源文件）
对应到 module 即可。

例：`7e26fb154 .. b5fceae83`(aiter HEAD) 区间改了 custom_all_reduce /
fused_qk_norm_rope_cache_quant / fused_split_gdr_update / asm_mla /
mla metadata，对应要删：

```bash
cd /app/aiter-test/aiter/jit
rm -f module_custom_all_reduce.so \
      module_fused_ar_mhc.so \
      module_fused_qk_norm_rope_cache_quant_shuffle.so \
      module_fused_split_gdr_update.so \
      module_mla_asm.so \
      module_mla_metadata.so \
      module_pa_metadata.so \
      module_ps_metadata.so
```

删后下次 launch 时按当前源码重编译（launch 会变慢）。

---

## 4. 启动 server（launch）

```bash
cd /app/ATOM
export AITER_BF16_FP8_MOE_BOUND=0
export ATOM_MOE_GU_ITLV=1
export NO_PROXY=localhost,127.0.0.1
export no_proxy=localhost,127.0.0.1
MODEL_DIR=/shared/data/amd_int/models/deepseek-ai/DeepSeek-V4-Pro

ENABLE_TORCH_PROFILER=0 ENABLE_RTL_PROFILER=0 \
  nohup .github/scripts/atom_test.sh launch "$MODEL_DIR" \
  --kv_cache_dtype fp8 -tp 8 --hf-overrides '{"use_index_cache": true, "index_topk_freq": 4}' \
  > /tmp/launch.log 2>&1 &
```

launch 会在 server 起来并 warmup 完成后才"算成功"。判据：
`/tmp/launch.log` 出现 `ATOM server warmup completed successfully`。
模型加载约 5 分钟；若删了 so 还要加编译时间。

内部真正执行的 server 命令：
```
python -m atom.entrypoints.openai_server --model $MODEL_DIR --server-port 8000 \
  --kv_cache_dtype fp8 -tp 8 --hf-overrides '{"use_index_cache": true, "index_topk_freq": 4}'
```

---

## 5. 跑 benchmark

warmup 完成后：

```bash
cd /app/ATOM
export AITER_BF16_FP8_MOE_BOUND=0
export ATOM_MOE_GU_ITLV=1
export NO_PROXY=localhost,127.0.0.1
export no_proxy=localhost,127.0.0.1
MODEL_DIR=/shared/data/amd_int/models/deepseek-ai/DeepSeek-V4-Pro

ENABLE_TORCH_PROFILER=0 \
RESULT_FILENAME=deepseek-v4-pro-1024-1024-32-0.8 \
SERVER_ARGS='--kv_cache_dtype fp8 -tp 8 --hf-overrides '"'"'{"use_index_cache": true, "index_topk_freq": 4}'"'"'' \
BENCH_EXTRA_ARGS='' \
ISL=1024 OSL=1024 CONC=32 RANDOM_RANGE_RATIO=0.8 \
  nohup .github/scripts/atom_test.sh benchmark "$MODEL_DIR" \
  > /tmp/bench.log 2>&1 &
```

内部真正执行的 benchmark 命令：
```
python -m atom.benchmarks.benchmark_serving \
  --model=$MODEL_DIR --backend=vllm --base-url=http://localhost:8000 \
  --dataset-name=random \
  --random-input-len=1024 --random-output-len=1024 --random-range-ratio=0.8 \
  --max-concurrency=32 --num-prompts=320 \
  --trust-remote-code --num-warmups=64 \
  --request-rate=inf --ignore-eos \
  --save-result --percentile-metrics=ttft,tpot,itl,e2el \
  --result-dir=. --result-filename=deepseek-v4-pro-1024-1024-32-0.8.json
```

结果表打印在 `/tmp/bench.log`，同时存 JSON 到
`/app/ATOM/<RESULT_FILENAME>.json`。

成功判据：`Successful requests: 320`（不是 0）。

---

## 6. 关闭 server

每次跑完用仓库脚本关：

```bash
cd /app/ATOM && bash scripts/stop_atom_server.sh
```

注意：该脚本有时第一次返回 exit 144（SIGTERM 后被中断），
**再跑一次** 即可；最后确认：

```bash
pgrep -af atom.entrypoints | grep -v grep || echo "无残留进程"
rocm-smi --showmemuse | grep "VRAM%"   # 应全为 0
```

---

## 7. 每卡吞吐换算

tp=8 是一个模型实例横跨 8 卡协同推理，单卡吞吐 = 聚合值 / 8（平均口径，
非每卡独立服务）：

- 每卡 output throughput = output_throughput / 8
- 每卡 total throughput  = total_token_throughput / 8

---

## 8. 已测结果记录（isl=1024 osl=1024 c=32）

| ATOM | aiter | Total tok/s | 每卡 total | Output tok/s | 每卡 output |
|---|---|---|---|---|---|
| 345d6a53 (HEAD) | b5fceae83 (HEAD) | 2901.16 | 362.6 | 1452.88 | 181.6 |
| 345d6a53 | 7e26fb154 | 2811.70 | 351.5 | 1408.08 | 176.0 |
| 345d6a53 | 7e26fb154 (清cache) | 2876.06 | 359.5 | 1440.31 | 180.0 |
| ea08015c | b5fceae83 (HEAD) | 2884.05 | 360.5 | 1444.31 | 180.5 |

观察：run-to-run 抖动约 ±3%，目前所有版本都落在 351~363/GPU，
尚未复现到 370 的版本。
