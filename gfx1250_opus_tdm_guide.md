# gfx1250 opus TDM 使用介绍

面向 aiter 仓库里 `csrc/include/opus/opus.hpp` **内置**的 `opus::tdm_desc` / `opus::tdm_window`
（gfx1250 / MI400 的 Tensor Data Mover）。这是重写后的新 API，和 `MI400 TDM 指令与 tdm_window
接入指南` HTML 一致。**不要**再用 `gcnasm/wmma_opus_rdna4` 里旧的 `TcopyDesc` / `SgprBitField`
（那是重写前的 API，已废弃）。

> 适用范围：所有定义都在 `#if defined(__gfx1250__) || !defined(__HIP_DEVICE_COMPILE__)` 里，
> 只在 gfx1250 device 编译 pass 和 host pass 存在；gfx942/gfx950 device pass 里 **不存在**。
> 因此凡是用到 `tdm_*` 的代码都必须自己再套一层 `#if defined(__gfx1250__)`。

---

## 1. TDM 是什么

Tensor Data Mover 是 gfx1250 上的专用 tensor DMA 单元，在 global memory 与 LDS 之间搬运
结构化 tile（最多 5D）。典型分工：一个 producer wave 发起一次 tile copy，consumer wave 之后
从 LDS 用 `ds_read` 喂给 WMMA/MFMA。

要点：
- 指令 `TENSOR_LOAD_TO_LDS`（global→LDS）/ `TENSOR_STORE_FROM_LDS`（LDS→global）。
- **wave 级 DMA，不是 per-lane**；**不受 EXEC 控制**（`EXEC==0` 也照发）。要让某些 wave 不发，
  必须用 `if (...)` 控制流包住 —— 通常只让一个 wave 发起整块 tile。
- OOB load 返回 0，OOB store 丢弃。
- 完成由 **TENSORcnt** 跟踪（和 loadcnt / asynccnt / dscnt 都不同）。
- 不自动和 LDS consumer 同步 —— 必须 `s_wait_tensorcnt` + workgroup barrier。

---

## 2. 两层封装

| 层 | 类型 | 角色 |
|---|---|---|
| Layer A | `opus::tdm_desc<T, TileDim0..4, flags...>` | 无状态 D#（`sg0[4]/sg1[8]/sg2[4]/sg3[4]` 裸 dword），编译期常量烧进初始化器，runtime 字段用 setter |
| Layer B | `opus::tdm_window<T, ..., CachePol>` | 在 tdm_desc 之上的**有状态**滑窗，`make` + `move(d0..d4, lds)` + `load_to_lds<cpol>()` |

模板参数（两者一致，顺序）：
```cpp
template<typename DataType,
         u64 TileDim0=0, TileDim1=0, TileDim2=0, TileDim3=0, TileDim4=0,
         u64 Count=1, GatherIndexSize=0, GatherMode=0, TypeLo=0, TypeHi=1,
         u64 AtomicBarrierEn=0, IterateEn=0, McEarlyTimeout=0, int SelectedWorkgroupCount=0,
         u64 LdsPadEn=0, PadInterval=0, PadAmount=0, typename SelectedWorkgroups=seq<>,
         int CachePol=default_cpol>   // 仅 tdm_window 有 CachePol
```
- `ndim` 由 TileDim 自动推：`TileDim4!=0?5 : TileDim3!=0?4 : TileDim2!=0?3 : 2`。
- `TileDim*` 是 tile 的**静态尺寸**（元素）；tensor 的边界（OOB 判断）是 runtime 的 `tensor_dim*`。

---

## 3. Cache 策略 (cpol)

```cpp
enum class opus::tdm_load_th : u8 { rt=0, nt=1, ht=2, bypass=3, nt_rt=4, rt_nt=5, nt_ht=6 };
enum class opus::tdm_scope   : u8 { cu=0, se=1, dev=2, sys=3 };
constexpr int opus::make_cpol(tdm_load_th th=rt, tdm_scope sc=dev, bool nv=false);
inline constexpr int opus::default_cpol = make_cpol(rt, dev);   // == 16
```
- 单 device GEMM 起点用 **16 (RT, DEV)**，靠 L2 LRU。
- 别无脑用 `sys`（旧默认 27，DF 流量浪费）。
- 流式只读用 `nt` / `bypass`（bypass 命中即丢，更激进）。
- `cpol` 必须是**编译期立即数**（模板参 / 字面量）。

---

## 4. 地址与 stride 语义（易错）

- **strides 单位是元素**，`global_addr` 是**字节地址**。硬件按 `data_size` 折算。
- `tensor_dim0_stride`(=make 的 `s0`) = 推进 **dim1** 的 stride（即 row stride）。
- `tensor_dim1_stride`(=make 的 `s1`) = 推进 **dim2** 的 stride（3D+ 用）。
- `tensor_dim2_stride`(=`s2`) = 推进 dim3；以此类推。
- **dim0 是连续内维**（元素 stride 隐含 1，无独立 stride 字段）。
- LDS 写入顺序：**dim0 最快 → dim1 → dim2 → …**。所以 3D tile 的 LDS 线性下标为
  `dim2*(td1*td0) + dim1*td0 + dim0`。
- LDS 地址在 gfx1250 是 **32-bit**：`static_cast<uint32_t>(reinterpret_cast<__UINTPTR_TYPE__>(smem_ptr))`。

`tdm_desc::make` 签名：
```cpp
void make(u32 lds_addr, const void* global_addr, u32 td0, u32 td1, u64 s0,
          u64 s1=0, u16 lds_bar=0, u32 td2=0, u32 td3=0, u64 s2=0, u64 s3=0, u32 td4=0);
```
注意实现里是 `if (td2) set_tensor_dim2(td2)` / `if (s1) set_tensor_dim1_stride(s1)` 等 ——
传 0 的维度不写（保持初始化器默认）。

---

## 5. 同步

TDM 完成 → **TENSORcnt**：
```cpp
opus::s_wait_tensorcnt(opus::number<N>{});   // 等 inflight TDM <= N；(1) 留 1 条做 pipeline，(0) 全清
```
gfx1250 是**分离计数器**，opus 提供合并 wrapper（`aiter_opus_plus.h`）：
```cpp
s_wait_all_loadcnt(number<load_cnt>, number<async_load_cnt>);  // gfx1250: 拆 s_wait_loadcnt + s_wait_asynccnt；gfx9: 合并 vmcnt
s_wait_all_dscnt(number<ds_cnt>);                               // gfx1250: s_wait_dscnt；gfx9: lgkmcnt
```
TENSORcnt 与 load/async/ds 都**独立**，所以 TDM 的等待要单独用 `s_wait_tensorcnt`。

典型 producer/consumer（同一 block 内所有 warp 既产又消费时）：
```cpp
// warp0 发起（EXEC 不 gate TDM，只让一个 wave 发）
if (warp_id == 0) win.load_to_lds();
...
// 消费前：
opus::s_wait_tensorcnt(number<keep>{});   // keep=1 循环内留下一条预取；keep=0 尾部 drain
__builtin_amdgcn_s_barrier();             // 跨 warp 发布 LDS（warp0 装的全部 head 对其它 warp 可见）
// ds_read + WMMA ...
```

> **inflight 上限**：机器约束 **最多 3 条 TDM inflight**（HTML 未写这条；HTML 只给了每 pipe
> 64×256B / 安全水位 ≤256KB 的容量限制）。双 buffer 预取 = 2 条 inflight，安全。

---

## 6. LDS padding

`LdsPadEn=1` 时每写够 `2^(PadInterval+1)` DWORD 跳过 `(PadAmount+1)*4 B` 空洞（防 bank
conflict）。consumer 端 row pitch 要用含 padding 的值（如 `Block_K+8`）。`PadInterval` 与
`PadAmount` 要么都 0 要么都非 0。连续无 padding 时全设 0（`LdsPadEn=0`）。

---

## 7. Cluster multicast（谨慎，违反即 hard hang）

`SelectedWorkgroups=seq<0,1>` + `SelectedWorkgroupCount=2` → 一次 global 读 multicast 到多个
WG 的 LDS。硬约束：
- **不能只选 1 个 WG**（mask popcount≤1 会被 `set_workgroup_mask` 强制写 0）。
- 必须 **cluster launch**；mask 选中的 WG 都必须真正 issue 对应 TDM。
- multicast **强制 force-miss WGP$**（无视 cpol 的 scope/th），必须满足 direct-copy
  （dim0 stride 256B/128B 对齐、不溢出 TCP），否则 hard hang。
- **per-WG 私有数据不要用 multicast**，`SelectedWorkgroups` 保持默认 `seq<>`。

---

## 8. 最小用法示例

### 2D（标准，用 tdm_window 全套：make + move + load_to_lds）
```cpp
#if defined(__gfx1250__)
using namespace opus;
using WinA = tdm_window<fp16_t, /*td0=*/Block_K, /*td1=*/16, 0,0,0,
                        1,0,0,0,1, 0,0,0, /*WgCount=*/0,
                        /*LdsPadEn*/1, /*PadInterval*/5, /*PadAmount*/3, seq<>>;
WinA win;
win.make(/*lds_base*/smembase, /*global_base*/ptr_a, /*lds_off*/0,
         /*td0*/K, /*td1*/Block_M, /*s0*/lda, /*o0*/0, /*o1*/wave_id*16);
win.load_to_lds();                                   // cpol = CachePol 默认 16
for (int ks = 0; ks < K_STEPS; ++ks) {
    if (ks < K_STEPS-1) s_wait_tensorcnt(number<1>{});
    else                s_wait_tensorcnt(number<0>{});
    __builtin_amdgcn_s_barrier();
    // ds_read + WMMA ...
    if (ks+2 < K_STEPS) { win.move(Block_K /*d0*/, 0_I,0_I,0_I,0_I, /*lds=*/dlds(ks)); win.load_to_lds(); }
}
#endif
```

### 3D（`tdm_window::make` 只有 2D → 手搓 desc，复用 load_to_lds）
```cpp
#if defined(__gfx1250__)
// 3D tile: dim0=inner(td0), dim1(td1, stride s0), dim2(td2, stride s1)
opus::tdm_window<T, td0, td1, td2> win;              // ndim=3（TileDim2!=0）
uint32_t lds = static_cast<uint32_t>(reinterpret_cast<__UINTPTR_TYPE__>(smem)) + slot_byte_off;
win.desc.make(lds, gptr, /*td0*/td0, /*td1*/td1, /*s0*/row_stride,
              /*s1*/dim2_stride, /*lds_bar*/0, /*td2*/td2);
win.load_to_lds();                                   // ndim==3 分支自动带 sg2/sg3
#endif
```
- 直接调 `win.load_to_lds()` 而**不要**自己写 `__builtin_amdgcn_tensor_load_to_lds`：5/6-operand
  差异（`OPUS_TDM_BUILTIN_HAS_SG_EXTRA`，按 clang 版本自动）在 load_to_lds 内部处理；且
  `OPUS_TDM_SG_EXTRA_ARG` 宏展开成裸 `impl::`，只在 `namespace opus` 内解析得到。
- 手填 `.desc` 后跳过了 window 的 `origin/extent` 状态，因此这种写法**不能再用 `win.move()`**；
  每步重新 `desc.make()` 即可。

---

## 9. 检查表

- [ ] 所有 `tdm_*` 用法套在 `#if defined(__gfx1250__)` 里。
- [ ] strides 用元素、global_addr 用字节、LDS addr 用 32-bit。
- [ ] 只让一个 wave 发起（`if (warp_id==0)`）；其它 warp 靠 `s_wait_tensorcnt`+`s_barrier` 看到数据。
- [ ] 用 `s_wait_tensorcnt`（不是 loadcnt/asynccnt）等 TDM；inflight ≤ 3。
- [ ] OOB：`tensor_dim1 = m_oob`（等）让越界行自动 load 0。
- [ ] per-WG 私有数据不开 multicast（`SelectedWorkgroups=seq<>`）。
- [ ] cpol 是编译期立即数，单 device 用 16。
- [ ] 3D 用 `desc.make()` 手搓 + `load_to_lds()`；不要手调 builtin。

---

## 参考
- `csrc/include/opus/opus.hpp`：`tdm_desc` / `tdm_window` / `make_cpol` 定义。
- `csrc/include/aiter_opus_plus.h`：`s_wait_all_loadcnt` / `s_wait_all_dscnt`。
- `op_tests/opus/device/test_tdm_gfx1250.cu`：官方 2D 2-slot ping-pong 例子。
- `tdm_instruction_guide 4.html`：位段 / cpol / multicast 细节（与上面一致）。
