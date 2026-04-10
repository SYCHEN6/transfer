# Ulysses Attention 通算掩盖优化方案

## 一、问题分析

### 原始代码执行时序

```
current_stream: [in_comm(0)] [attention(0)] [wait] [attention(1)] [wait] [attention(2)] ...
stream2:        [等HCCL释放] [in_comm(1)] [in_comm(2)] ... [in_comm(N-1)] [out_comm(0→N)]
```

原始代码存在以下四个关键问题：

**问题 1：in_comm(0) 放在 current_stream 上**

HCCL 对同一 process group 的 collective 操作会跨流串行化。`in_comm(0)` 在 `current_stream`，`in_comm(1)` 在 `stream2`，HCCL 强制 `in_comm(1)` 等待 `in_comm(0)` 完成后才能执行，导致 stream2 的启动被推迟。

**问题 2：所有 in_comm 一次性全部提交给 stream2**

Python 在 `with torch.npu.stream(self.stream2)` 块中把 `in_comm(1..N-1)` 全部提交完毕后，`current_stream` 才开始做 attention。此时 in_comm 们已经在排队甚至完成，真正的通算重叠窗口消失或极短。

**问题 3：out_comm 全部堆在末尾，无法与 attention 重叠**

output AllToAll 的 `with stream2` 块在主计算循环之后才提交，导致 stream2 的队列是：
```
[in_comm(1)] → [in_comm(2)] → ... → [in_comm(N-1)] → [out_comm(0)] → [out_comm(1)] → ...
```
out_comm 无法与任何 attention 计算重叠。

**问题 4：`self.event[i]` 被两个流重复 record，存在竞态**

- stream2 在 `in_comm(i)` 后 record `event[i]`（含义：通信完成）
- current_stream 在 `attention(i)` 后再次 record `event[i]`（含义：计算完成）
- output 段的 `stream2.wait_event(event[i])` 等的是哪次 record 取决于时序，行为未定义

---

## 二、优化方案

### 核心思路：双缓冲 + 流水线交错

**目标时序：**

```
current_stream: ──[attention(0)]──────[attention(1)]──────[attention(2)]──────[attention(3)]──
stream2:        [QKV(0)][QKV(1)][out(0)][QKV(2)][out(1)][QKV(3)][out(2)][out(3)]
                          ↑重叠↑          ↑重叠↑           ↑重叠↑
```

### 改动一：所有 in_comm 移到 stream2

将 chunk 0 的 QKV AllToAll 从 `current_stream` 移到 `stream2`，作为预取（Prefetch）。`current_stream` 专职做 attention 计算，不再参与通信。

### 改动二：独立 Event，消除竞态

用三组独立 event 替换复用的 `self.event[i]`：

| Event | 含义 | 由谁 record |
|---|---|---|
| `in_events[i]` | in_comm(i) 完成 | stream2 |
| `comp_events[i]` | attention(i) 完成 | current_stream |
| `out_events[i]` | out_comm(i) 完成 | stream2 |

### 改动三：循环内交错提交，形成正确的流水线队列

在主循环的每次迭代 `i` 中，按以下顺序向两个流提交任务：

1. 向 stream2 提交 `in_comm(i+1)`（非阻塞，立即返回）
2. current_stream 等待 `in_events[i]`，执行 `attention(i)` 并 record `comp_events[i]`
3. 向 stream2 提交 `out_comm(i)`（带 `wait(comp_events[i])`）

Python 提交极快（近乎瞬间），因此 stream2 的硬件队列最终形成以下交错结构：

```
stream2 实际硬件队列：
QKV(0) → QKV(1) → wait(comp[0]) → out(0) → QKV(2) → wait(comp[1]) → out(1) → QKV(3) → ...
```

### 改动四：移除无效的 qkv_event

原始代码中 `qkv_event` 在任何通信提交之前就 record，`stream2.wait_event(qkv_event)` 几乎立即满足，且放在循环内每次都 wait，完全无效。优化后直接移除。

---

## 三、修改前后对比

| 维度 | 原始代码 | 优化后 |
|---|---|---|
| in_comm(0) 的流 | `current_stream` | `stream2`（预取） |
| in_comm 提交方式 | 全部一次性提交给 stream2 | 循环内逐步提交（双缓冲） |
| out_comm 位置 | 全部堆在计算循环结束后 | 循环内与下一块 attention 重叠 |
| Event 使用 | `self.event[i]` 两流复用（竞态） | 三组独立 event |
| qkv_event | 无效的 wait，循环内重复 | 已移除 |
| attention(i) 与 in_comm(i+1) | 部分重叠（依赖时序巧合） | 结构保证重叠 |
| out_comm(i) 与 attention(i+1) | 无重叠 | 结构保证重叠 |

---

## 四、完整修改后代码

```python
query_layer_list = query.split(self.ulysses_world_size, dim=2)  # N/(tp*sp) * B, S/sp, sp, D
key_layer_list   = key.split(self.ulysses_world_size, dim=2)
value_layer_list = value.split(self.ulysses_world_size, dim=2)
for_loop = len(query_layer_list)

# 预分配缓冲区
q_bufs     = [None] * for_loop
k_bufs     = [None] * for_loop
v_bufs     = [None] * for_loop
output_fa  = [None] * for_loop
output_res = [None] * for_loop

# 三组独立 event，避免复用引发的竞态条件
in_events   = [torch.npu.Event() for _ in range(for_loop)]  # in_comm(i) 完成
comp_events = [torch.npu.Event() for _ in range(for_loop)]  # attention(i) 完成
out_events  = [torch.npu.Event() for _ in range(for_loop)]  # out_comm(i) 完成

def submit_in_comm(i):
    """在当前激活的 stream 上提交第 i 块的 QKV AllToAll + ring gather + joint cat"""
    q = SeqAllToAll4D.apply(self.ulysses_pg, query_layer_list[i], self.scatter_idx, self.gather_idx)
    k = SeqAllToAll4D.apply(self.ulysses_pg, key_layer_list[i],   self.scatter_idx, self.gather_idx)
    v = SeqAllToAll4D.apply(self.ulysses_pg, value_layer_list[i], self.scatter_idx, self.gather_idx)

    if self.ring_world_size > 1:
        b, s, n, d = k.shape
        k_full = torch.empty([self.ring_world_size, b, s, n, d], dtype=key.dtype, device=key.device)
        v_full = torch.empty([self.ring_world_size, b, s, n, d], dtype=value.dtype, device=value.device)
        dist.all_gather_into_tensor(k_full, k, group=self.ring_pg)
        dist.all_gather_into_tensor(v_full, v, group=self.ring_pg)
        k = k_full.permute(1, 0, 2, 3, 4).reshape(b, -1, n, d)
        v = v_full.permute(1, 0, 2, 3, 4).reshape(b, -1, n, d)

    if is_joint:
        start = self.ulysses_rank + self.ulysses_world_size * i
        jk = joint_tensor_key_all[..., start:start + 1, :]
        jv = joint_tensor_value_all[..., start:start + 1, :]
        k = torch.cat([k, jk], dim=1)
        v = torch.cat([v, jv], dim=1)

    q_bufs[i], k_bufs[i], v_bufs[i] = q, k, v

# ── 预取 chunk 0（Prefetch）─────────────────────────────────────────────────
# 改动：原先 chunk 0 的 AllToAll 在 current_stream，现在移到 stream2
with torch.npu.stream(self.stream2):
    submit_in_comm(0)
    in_events[0].record(self.stream2)

# ── 主流水线循环 ──────────────────────────────────────────────────────────────
# Python 提交几乎瞬间完成，stream2 的硬件队列形成交错结构：
#   QKV(0) → QKV(1) → wait(comp[0]) → out(0) → QKV(2) → wait(comp[1]) → out(1) → ...
# 从而保证：
#   attention(i) 与 in_comm(i+1) 重叠
#   out_comm(i)  与 attention(i+1) 重叠
for i in range(for_loop):
    # 步骤 1：向 stream2 提交 in_comm(i+1)（非阻塞，立即返回）
    #         此时 stream2 会在处理完 in_comm(i) 后立即开始 in_comm(i+1)
    if i + 1 < for_loop:
        with torch.npu.stream(self.stream2):
            submit_in_comm(i + 1)
            in_events[i + 1].record(self.stream2)

    # 步骤 2：current_stream 等待 in_comm(i) 完成，执行 attention(i)
    #         此时 stream2 正在执行 in_comm(i+1) → 真正的通算重叠
    self.current_stream.wait_event(in_events[i])
    if self.algo == 1:
        output_fa[i] = attention_LA(q_bufs[i], k_bufs[i], v_bufs[i], scale)
    elif self.algo == 0:
        output_fa[i] = attention_FAScore(q_bufs[i], k_bufs[i], v_bufs[i], scale)
    elif self.algo == 2:
        output_fa[i] = self.quant_fa(q_bufs[i], k_bufs[i], v_bufs[i])
    else:
        raise ValueError(f"select flash attention algorithm only support 0, 1, 2, but got {self.algo}")
    # 显式指定流，避免多流环境下 record 到错误的流
    comp_events[i].record(self.current_stream)

    # 步骤 3：向 stream2 提交 out_comm(i)
    #         stream2 会等 attention(i) 完成（wait comp_events[i]）后执行
    #         out_comm(i) 排在 in_comm(i+1) 之后、in_comm(i+2) 之前
    #         → out_comm(i) 与 attention(i+1) 重叠
    with torch.npu.stream(self.stream2):
        self.stream2.wait_event(comp_events[i])
        output_res[i] = SeqAllToAll4D.apply(
            self.ulysses_pg, output_fa[i], self.gather_idx, self.scatter_idx
        )
        out_events[i].record(self.stream2)

# ── 将输出同步回 current_stream ──────────────────────────────────────────────
for i in range(for_loop):
    self.current_stream.wait_event(out_events[i])

output = torch.cat(output_res, dim=2)
```

---

## 五、流水线时序详解（以 for_loop=4 为例）

### stream2 硬件队列（Python 提交完成后）

```
[Prefetch] QKV(0) → in_ev[0]
[i=0]      QKV(1) → in_ev[1]
[i=0]      wait(comp[0]) → out(0) → out_ev[0]
[i=1]      QKV(2) → in_ev[2]
[i=1]      wait(comp[1]) → out(1) → out_ev[1]
[i=2]      QKV(3) → in_ev[3]
[i=2]      wait(comp[2]) → out(2) → out_ev[2]
[i=3]      wait(comp[3]) → out(3) → out_ev[3]
```

### current_stream 硬件队列

```
wait(in_ev[0]) → attention(0) → comp_ev[0]
wait(in_ev[1]) → attention(1) → comp_ev[1]
wait(in_ev[2]) → attention(2) → comp_ev[2]
wait(in_ev[3]) → attention(3) → comp_ev[3]
```

### 实际 GPU 执行时序

```
时间轴 →
current_stream:  │◄── attention(0) ──►│◄── attention(1) ──►│◄── attention(2) ──►│◄── attention(3) ──►│
stream2:         │QKV0│QKV1│   out0   │QKV2│   out1   │QKV3│   out2   │   out3   │
                       ↑               ↑               ↑
                    重叠区            重叠区            重叠区
```

- **QKV(1)** 与 **attention(0)** 重叠
- **out(0)** + **QKV(2)** 与 **attention(1)** 重叠（若 out(0)+QKV(2) 总耗时 ≤ attention(1)，则全程掩盖）
- **out(1)** + **QKV(3)** 与 **attention(2)** 重叠
- **out(2)** 与 **attention(3)** 重叠
- **out(3)** 无法掩盖（最后一个操作，无后续计算）

### HCCL 串行化说明

HCCL 对同一 process group（`ulysses_pg`）的 collective 操作跨流串行化，因此 in_comm 和 out_comm 即使在同一个 stream2 上也是串行的。优化的收益来自：**通信（stream2）与计算（current_stream）使用不同硬件资源**，二者可以真正并行。

---

## 六、注意事项

1. **`submit_in_comm` 内的张量写入**：`q_bufs[i]`、`k_bufs[i]`、`v_bufs[i]` 由 stream2 写入，current_stream 通过 `wait_event(in_events[i])` 保证可见性，无需额外同步。

2. **`comp_events[i].record(self.current_stream)`**：必须显式传入流参数，防止在多流切换场景下 record 到错误的流。

3. **ring_pg 与 ulysses_pg 的关系**：若二者为不同 process group，ring all-gather 与 ulysses AllToAll 之间不存在 HCCL 串行化约束，重叠效果更好；若为同一 group，则仍串行，但优化结构不变。

4. **内存峰值**：双缓冲要求同时持有 `q/k/v_bufs` 的多个 chunk，显存占用略有上升，需根据实际显存余量评估。
