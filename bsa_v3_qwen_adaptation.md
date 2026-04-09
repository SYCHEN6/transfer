# Qwen-Image-Edit-2509 BSA v3 适配说明

## 背景

Wan2.2 通过 MindIE-SD 的 `bsa_sparse_attention_v3` 接口接入 Block Sparse Attention（BSA），在长序列视频推理上获得了显著加速。Qwen-Image-Edit-2509 是图像编辑模型，joint attention 结构与 Wan 相似，但存在以下关键差异，需要专项适配。

---

## 一、Token 顺序：[txt, img] vs [img, txt]

### 原理

`bsa_sparse_attention_v3` 内部调用 `do_tensor_rearrange_pooling`（来自 rf_v2），该函数的 token 布局约定是：

```
输入: [txt_tokens, img_tokens]   ← txt 在前
处理: 仅对 img 部分做空间块重排，txt 保持原序
输出: [img_rearranged, txt]      ← 内部转换为 img 在前供 BSA 算子使用
```

`do_tensor_inv_rearrange` 负责逆向恢复：

```
输入: [img_rearranged, txt]      ← BSA 算子输出
输出: [txt, img_original]        ← 还原为 txt 在前
```

因此 `bsa_sparse_attention_v3` 的 **输入和输出都是 `[txt, img]` 顺序**。

### Qwen 的情况

Qwen 的 joint attention 天然就是 `[txt, img]` 顺序（encoder_hidden_states 在前，hidden_states 在后）：

```python
joint_query = torch.cat([txt_query, img_query], dim=1)  # [txt, img] ✓
```

这与 `bsa_sparse_attention_v3` 期望的输入顺序完全一致，**无需任何 reorder**，输出也直接按 `seq_txt` 切分即可。

### 曾走过的弯路

早期代码误以为"BSA 期望 txt 在末尾"，在调用前做了额外的 reorder：

```python
# 错误做法（增加了 6 次多余 ConcatD）
_q = torch.cat([joint_query[:, txt_len:], joint_query[:, :txt_len]], dim=1)  # [txt,img] → [img,txt]
```

这导致函数把前 `txt_len` 个 **img token** 当成文本保护区，把真正的 txt token 当成图像 token 做空间重排，语义完全错误。即使没有 crash，输出质量也会下降。

### 最终做法

```python
# 正确做法：joint_query 已经是 [txt, img]，直接传入
joint_hidden_states, new_mask = bsa_sparse_attention_v3(
    joint_query, joint_key, joint_value,
    txt_len=txt_query.shape[1],
    ...
)
# 输出也是 [txt, img]，直接按 seq_txt 切分
txt_attn_output = joint_hidden_states[:, :seq_txt, :]
img_attn_output = joint_hidden_states[:, seq_txt:, :]
```

---

## 二、latent_shape 的多图拼接问题

### 原理

`bsa_sparse_attention_v3` 需要 `latent_shape_q = (t, h, w)` 描述图像 token 的空间结构，以便正确执行块重排。其中 `t * h * w` 必须等于实际的 img token 数。

### 问题

`grid_sizes` 由 transformer forward 构造时只取了每个 batch 第一张图的 shape：

```python
grid_sizes_list = [shape[0] for shape in img_shapes]  # 只取第一张图！
```

Qwen 图像编辑场景中，模型同时处理**源图 + 参考图**（或多图拼接），`img_query.shape[1]` 是所有图 token 之和（如 2 × 4096 = 8192），而 `grid_sizes[0]` 给出的是单图维度 `(1, 64, 64)` → 4096，不匹配。

传入错误 `latent_shape` 后，`rearrange_with_remaining` 计算 token 数时会崩溃：

```
Shape mismatch, 8192 != 4096
```

### 修复

在 processor 内用实际 img token 数反推总帧数：

```python
if grid_sizes is not None:
    _h = int(grid_sizes[0][1])
    _w = int(grid_sizes[0][2])
    _single = int(grid_sizes[0][0]) * _h * _w   # 单图 token 数，如 1×64×64=4096
    _num_imgs = img_query.shape[1] // _single    # 实际图数，如 8192//4096=2
    latent_shape = (_num_imgs, _h, _w)           # (2, 64, 64) ✓
```

这样即使拼接图数变化，也能自动适配，不依赖外部 `grid_sizes` 的准确性。

---

## 三、MindIE-SD 逆重排 txt_len 修复

### 原理

`bsa_sparse_attention_v3` 内部的逆重排函数 `_bsa_inv_rearrange` 是为 Wan（纯图像，txt_len=0）设计的，不接受 `txt_len` 参数。当 `txt_len > 0`（joint attention）时，它会尝试把整个输出序列（包含 txt token）当成 img token 做逆重排，导致 reshape 报错：

```
RuntimeError: shape '[1, 0, 8, 8, 8, 8, 24, 128]' is invalid for input of size 13289472
```

### 修复

在 `sparse_flash_attn_rf_v3.py` 的 `bsa_sparse_attention_v3` 末尾，根据 `txt_len` 分支选择逆重排函数：

```python
# 修复前：总是调用 _bsa_inv_rearrange（不支持 txt_len）
out = _bsa_inv_rearrange(out, tq, hq, wq)

# 修复后
if txt_len > 0:
    # joint attention：使用 do_tensor_inv_rearrange（来自 rf_v2，正确处理 txt/img 分离）
    out = do_tensor_inv_rearrange(out, txt_len, latent_shape_q, latent_shape_k, input_layout)
else:
    # 纯图像（Wan）：保留原始逻辑，包含首帧特殊处理
    out = _bsa_inv_rearrange(out, tq, hq, wq)
```

`do_tensor_inv_rearrange` 从 rf_v2 导入，已在文件顶部 import，无需额外依赖。

> **注意**：本地 MindIE-SD 源码的改动**不影响运行时**，因为运行时加载的是 site-packages 中安装的版本。需要将此修复同步到环境机器上的安装包：
> ```
> /home/miniconda3/envs/csy_bsa/lib/python3.11/site-packages/mindiesd/layers/flash_attn/sparse_flash_attn_rf_v3.py
> ```

---

## 四、t_idx：去噪步索引 vs 块索引

### 原理

`rainfusion_config["skip_timesteps"]` 表示**前 N 个去噪步**使用 dense attention（噪声大、注意力分布均匀，稀疏效果差），从第 N 步起启用 BSA。

mask 缓存逻辑也依赖去噪步索引：

```python
(t_idx - skip_ts) % mask_refresh_interval != 0  # 判断本步是否需要刷新 mask
```

### 问题

块循环里错误地把**块索引**（0~59）当成 `t_idx` 传入：

```python
for index_block, block in enumerate(self.transformer_blocks):
    block(..., t_idx=index_block, ...)  # 错误：块索引，不是去噪步索引
```

后果：
- `skip_timesteps=15` 实际变成"第 15 号之后的块用稀疏"，每个去噪步都有 45/60 的块走 BSA
- mask 缓存永远不生效：每个块的 `t_idx` 固定（不随去噪步变化），`(t_idx - skip_ts) % 1 = 0` 永远为真，永远重新计算 mask

### 修复

三处改动联动：

**1. `transformer.forward()` 增加参数**

```python
def forward(self, ..., t_idx: int = None, ...):
```

**2. 块循环改用 `t_idx`**

```python
# 修复前
block(..., t_idx=index_block, ...)

# 修复后
block(..., t_idx=t_idx, ...)
```

**3. pipeline 在去噪循环里传入步骤索引**

```python
for i, t in enumerate(timesteps):
    noise_pred = self.transformer(..., t_idx=i, ...)
```

修复后，`skip_timesteps=15` 表示前 15 个**去噪步**用 dense，第 15 步起所有块均使用 BSA；mask cache 也能按 `mask_refresh_interval` 正常控制刷新频率。

---

## 五、mask 缓存（mask_refresh_interval）

### 原理

BSA 每步都需要生成稀疏掩码，流程为：

1. **avgpool**（~746µs）：对 q/k/v 做平均池化，降低计算量
2. **mask 生成**（~117µs）：计算 attention score，取 topk，生成 int8 二值掩码

两步合计约 **863µs/步**，对长推理影响显著。图像内容在去噪过程中变化缓慢，掩码可以跨步复用。

### 配置

```python
rainfusion_config = {
    "mask_refresh_interval": 0,  # 0=永久复用（最快），N=每N步刷新
    ...
}
```

### 实现

processor 上保存 `self._bsa_mask_cache`，跨去噪步持久化：

```python
# 判断是否复用缓存
if self._bsa_mask_cache is not None and (
    mask_refresh == 0 or
    (t_idx - skip_ts) % mask_refresh != 0
):
    use_cached = True   # 跳过 avgpool + mask 生成，节省 ~863µs

# 早期步（t_idx < skip_ts）清除缓存
self._bsa_mask_cache = None
```

`bsa_sparse_attention_v3` 的 `cached_mask` 参数：非 None 时内部走 `do_tensor_rearrange_only`（只做空间重排，跳过 avgpool）。

---

## 六、latent_shape 说明

| 模型 | 分辨率 | VAE 下采样 | Qwen 2×2 打包 | img token 数 | latent_shape |
|------|--------|-----------|--------------|-------------|-------------|
| 单图编辑（1024×1024） | 1024×1024 | ÷8 → 128×128 | ÷2 → 64×64 | 4096 | (1, 64, 64) |
| 双图拼接（源图+参考图） | 同上 | 同上 | 同上 | 8192 | **(2, 64, 64)** |

`latent_shape` 中的 `t`（帧数/图数）需要与实际 img token 数匹配，不能硬编码为 1。

---

## 七、首帧保护机制与 Qwen 的适配

### 首帧保护是什么

BSA 在生成稀疏 mask 时，除了用 topk 选出最相关的 block 之外，还会强制让某些 block 保持全连接（即 mask 值为 1）。其中一个机制叫**首帧保护**：把空间序列中"第一帧"对应的所有行和列都强制设为 1。

用一个简单的比喻：mask 是一张"谁可以看谁"的通行证矩阵。行代表"查询方"，列代表"被查询方"。强制某行为 1 = 该 token 可以看所有人；强制某列为 1 = 所有人都可以看该 token。

```
mask 矩阵示意（■=1允许关注，□=0跳过）：

         被查询方 →
         [首帧] [其他img] [txt]
查  [首帧]  ■■■■   ■■■■■■   ■■      ← 整行强制 1
询  [其他]  ■■■■   □■□■□□   ■■      ← 首帧列强制 1，其余 topk
方  [txt]   ■■■■   ■■■■■■   ■■      ← 整行强制 1
↓
```

首帧整行和整列都是 1，形成一个"L 形"或"十字形"密集区域。

### 为什么 Wan 视频需要首帧保护

Wan 是视频生成模型，输入是多帧序列（如 16 帧）：

```
帧序列：[帧0（参考帧）] [帧1] [帧2] ... [帧15]
          ↑ 首帧 = 参考帧，内容最确定、质量最高
```

- **帧 0（参考帧）**：由用户提供或去噪最充分，作为其他帧的内容锚点
- **其他帧**：需要在生成时持续"参考"帧 0，保持内容一致性

因此需要保证"所有帧都能无条件关注帧 0"（强制帧 0 的列为 1），同时帧 0 本身也应保持全局感知（强制帧 0 的行为 1）。

**帧 0 的 token 占比：** `first_frame_len / total_img_tokens = h*w / (t*h*w) = 1/t`

| Wan 帧数 | 首帧 block 占比 |
|---------|----------------|
| 4 帧    | 1/4 = 25%      |
| 8 帧    | 1/8 = 12.5%    |
| 16 帧   | 1/16 = **6.3%** |

帧数越多，首帧占比越小，"L 形"密集区对整体稀疏度的影响越小。

### 为什么 Qwen 图像编辑不需要首帧保护

Qwen 是**图像编辑**模型，输入是 2 张图拼接（源图 + 参考图）：

```
图像序列：[源图（图0）] [目标图（图1）]
latent_shape = (t=2, h=64, w=64)
```

此时 `first_frame_len = h * w = 64 * 64 = 4096`，正好是**一张图的全部 token**。

**首帧 block 占比：** `4096 / 8192 = 50%`

这意味着强制保护把整个矩阵的一半行和一半列都设为 1，形成一个极大的"L 形"密集区：

```
Qwen mask 矩阵（64 img块 + 4 txt块 = 68块，protect_first_frame=True）：

         col:  [0  ...  31]  [32  ...  63]  [64..67]
               (图0，32块)   (图1，32块)    (txt，4块)

row 0-31  ■■■■■■■■■■■■■■■■  ■■■■■■■■■■■■■■■■  ■■■■   ← 整行强制1（图0查询方）
(图0)

row 32-63 ■■■■■■■■■■■■■■■■  □■□□■□□■□□■□□■□  ■■■■   ← 图0列强制1，其余topk
(图1)

row 64-67 ■■■■■■■■■■■■■■■■  ■■■■■■■■■■■■■■■■  ■■■■   ← 整行强制1（txt查询方）
(txt)
```

**真正受 topk 控制的只有右上角 32×32 的小方块**（图1 查询 × 图1 被查询），
仅占矩阵总面积的 `32×32 / 68×68 ≈ 22%`，其中 50% 是稀疏的，
所以整体稀疏度只有 `22% × 50% ≈ 11%`。

对比关闭首帧保护后（`protect_first_frame=False`）：

```
Qwen mask 矩阵（protect_first_frame=False）：

         col:  [0  ...  63]  [64..67]
               (img，64块)   (txt，4块)

row 0-63  □■□□■□□■□□■□■□□  ■■■■   ← 仅 topk 控制，txt列强制1
(img)

row 64-67 ■■■■■■■■■■■■■■■■  ■■■■   ← txt整行强制1
(txt)
```

**受 topk 控制的区域变为 64×64**，占比从 22% 提升到 `64×64 / 68×68 ≈ 88.7%`，
整体稀疏度接近 `88.7% × 50% ≈ 44%`，接近配置值 50%。

### 两种场景对比总结

|  | Wan 视频（16帧） | Qwen 图像编辑（2图） |
|--|----------------|---------------------|
| 首帧语义 | 参考帧，其他帧需要持续对齐 | 源图，地位与目标图对等 |
| 首帧 token 占比 | 1/16 ≈ 6% | 1/2 = **50%** |
| 强制 dense 区域 | 矩阵边缘窄边 | **矩阵一半行和一半列** |
| 首帧保护后稀疏度 | ~45%（影响很小） | **~11%（稀疏度严重压缩）** |
| 是否需要首帧保护 | ✅ 需要 | ❌ 不需要 |

### 修复

在 `bsa_sparse_attention_v3` 加 `protect_first_frame` 参数（默认 `True` 保持 Wan 兼容），Qwen 侧传 `False`：

```python
# get_blockwise_mask_binary（MindIE-SD）
if protect_first_frame:
    firstframe_block_num = (first_frame_len + pool_size - 1) // pool_size
    if firstframe_block_num > 0:
        mask[:, :, :firstframe_block_num, :] = True
        mask[:, :, :, :firstframe_block_num] = True

# transformer_qwenimage.py（Qwen 侧）
joint_hidden_states, new_mask = bsa_sparse_attention_v3(
    ...,
    protect_first_frame=False,  # 关闭首帧保护，稀疏度从 ~11% 恢复到 ~45%
)
```

**效果：** mask 稀疏度从 ~11% 提升到 ~45%，BSA 有效加速，实测性能提升显著。

---

## 九、文件改动清单

### `MindIE-SD/mindiesd/layers/flash_attn/sparse_flash_attn_rf_v3.py`

- **逆重排分支**：`txt_len > 0` 时改用 `do_tensor_inv_rearrange`，避免 joint attention 场景下 reshape 崩溃

> 同步路径：需同步到 site-packages 安装包

### `Qwen-Image-Edit-2509/qwenimage_edit/transformer_qwenimage.py`

| 位置 | 改动 | 原因 |
|------|------|------|
| `QwenDoubleStreamAttnProcessor2_0.__init__` | 加 `self._bsa_mask_cache = None` | 跨步复用 mask |
| `__call__` joint 拼接 | `cat([txt_query, img_query])` 不做 reorder | txt 在前符合 BSA pipeline 期望 |
| `__call__` latent_shape 计算 | 从 img_query token 数反推 `_num_imgs` | 支持多图拼接 |
| `__call__` BSA 调用 | 直接传 `joint_query`，`txt_len=txt_query.shape[1]` | 消除多余 reorder ConcatD |
| `__call__` 输出切分 | `joint[:, :seq_txt]` / `joint[:, seq_txt:]` | 输出已是 [txt, img] 顺序 |
| `forward()` 签名 | 加 `t_idx: int = None` | 接收去噪步索引 |
| 块循环 | `t_idx=t_idx`（原为 `index_block`） | 修复步索引 vs 块索引混淆 |

### `Qwen-Image-Edit-2509/qwenimage_edit/pipeline_qwenimage_edit_plus.py`

- 所有 `self.transformer(...)` 调用加 `t_idx=i`（`i` 为去噪循环计数器）

### `Qwen-Image-Edit-2509/run_edit_2509.py` / `run_edit_2509_cfg_usp.py`

- 新增 `--mask_refresh_interval` 参数（默认 1）
- `rainfusion_type` choices 限定为 `["v3"]`
- `rainfusion_config` 中加入 `mask_refresh_interval` 字段

---

## 十、性能说明

### 优化前后对比

| 阶段 | 配置 | 耗时 | 实际 mask 稀疏度 |
|------|------|------|-----------------|
| 基线（纯 FA） | — | ~67s | — |
| BSA 初版 | sparsity=0.5，protect_first_frame=True | ~84s（劣化） | **~11%** |
| BSA 优化版 | sparsity=0.5，protect_first_frame=False | **提升** | **~45%** |

### 初版劣化的三个原因

1. **有效稀疏度仅 11%**：首帧保护强制覆盖 50% 的 block，topk 几乎没有发挥空间（本章节详细分析见第七章）
2. **掩码开销未优化**：`mask_refresh_interval=1` 时每步都重新计算掩码（+863µs/块）
3. **序列偏短**：BSA 的硬件优势在超长序列（>16K token）时才明显，当前 ~8700 tokens 时 BSA 算子本身比标准 FA 慢

### 后续建议

- `mask_refresh_interval=0`（永久复用 mask）可进一步节省每步 ~863µs × 块数 的掩码生成开销
- BSA 更适合多图并行、更高分辨率或视频推理等超长序列场景（>16K tokens）
