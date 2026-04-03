# Qwen-Image-Edit-2509 稀疏 FA v3 适配记录

**日期：** 2026-04-03  
**目标：** 参考 Wan2.2 的 rainfusion 实现，为 Qwen-Image-Edit-2509 接入 MindIE 的 Block Sparse Attention（BSA）v3 加速。

---

## 一、背景与差异分析

### Wan2.2（参考实现）
- **模型类型：** 视频生成，latent 有时间维度 `(T, H, W)`
- **attention 结构：** 图像 token 和文本 token **分开处理**，稀疏注意力仅作用于图像 token
- **token 顺序：** `[img, txt]`（文本在末尾）
- **latent_shape：** 由 `grid_sizes` 从 patch embedding 后的空间维度直接获取

### Qwen-Image-Edit-2509（适配目标）
- **模型类型：** 图像编辑，latent 为 2D，经 2×2 packing 后 `H_pack = H_latent // 2`
- **attention 结构：** **Joint attention**，文本和图像 token 拼接后一起计算，顺序为 `[txt, img]`
- **token 数量：** 以 1024×1024 单图为例，`img_tokens = 64×64 = 4096`，`txt_tokens ≈ 4326`，总计 `~8422`
- **latent_shape 来源：** pipeline 构造的 `img_shapes`，格式为 `[[batch0: (1,H_pack,W_pack), ...], ...]`，`//2` 已在 pipeline 中完成

---

## 二、修改文件清单

| 文件 | 修改内容 |
|------|---------|
| `run_edit_2509.py` | 新增 `add_rainfusion_args()`；`_parse_args()` 中调用；加载 transformer 后设置 `transformer.rainfusion_config` |
| `run_edit_2509_cfg_usp.py` | 同上；`parallel_forward` 增加 `t_idx` 参数和 `effective_attn_kwargs` 注入；`xFuserQwenDoubleStreamAttnProcessor.__call__()` 接受新 kwargs |
| `qwenimage_edit/pipeline_qwenimage_edit_plus.py` | 去噪循环 4 处 `self.transformer(...)` 调用均增加 `t_idx=i` |
| `qwenimage_edit/transformer_qwenimage.py` | 核心改动，见下节 |

---

## 三、核心改动详解（transformer_qwenimage.py）

### 3.1 参数入口

```python
def add_rainfusion_args(parser):
    group = parser.add_argument_group(title="Rainfusion args")
    group.add_argument("--use_rainfusion", action='store_true')
    group.add_argument("--sparsity", type=float, default=0.64)
    group.add_argument("--sparse_start_step", type=int, default=15)
    return parser

# main() 中加载 transformer 后：
if args.use_rainfusion:
    transformer.rainfusion_config = {
        "sparsity": args.sparsity,
        "skip_timesteps": args.sparse_start_step,
        "type": "v3",
    }
```

### 3.2 latent_shape 推导

```
img_shapes 结构：
  [[batch0: (1, H_pack, W_pack), (1, H_pack, W_pack), ...], [batch1: ...], ...]
              ↑ 每张图一个 tuple，pipeline 已做 // vae_scale_factor // 2

latent_shape = (num_imgs, H_pack, W_pack)
  num_imgs = len(img_shapes[0])   # 图片数，对应 Wan 的帧数 T
  H_pack   = img_shapes[0][0][1] # 已是 packed 维度，无需再 //2
  W_pack   = img_shapes[0][0][2]
```

对应 Wan2.2 的等价关系：

| Wan2.2 | Qwen |
|--------|------|
| `grid_sizes[b]` = `(T_pat, H_pat, W_pat)` | `latent_shape` = `(num_imgs, H_pack, W_pack)` |
| patch embedding 后维度 | pipeline 构造时已完成 2x2 packing |

### 3.3 Joint Attention 的 token 顺序处理

Qwen 拼接顺序为 `[txt, img]`，BSA 函数期望 `[img, txt]`（text 在末尾由 mask 保护）。

```python
img_token_num = latent_shape[0] * latent_shape[1] * latent_shape[2]
txt_len = joint_query.shape[1] - img_token_num  # 总长 - 图像 token 数

# 重排为 [img, txt]
sparse_q = torch.cat([joint_query[:, txt_len:], joint_query[:, :txt_len]], dim=1)
# k, v 同理

# ... 稀疏注意力计算 ...

# 输出还原为 [txt, img]
joint_hidden_states = torch.cat([out[:, img_token_num:], out[:, :img_token_num]], dim=1)
```

### 3.4 稀疏注意力调用（绕过 bsa_sparse_attention_v3 的 bug）

#### 为什么需要绕过？

安装版 MindIE 的 rf_v3 调用链如下：

```
sparse_attention(sparse_type="rf_v3")          # sparse_flash_attn.py line 139
  └─ bsa_sparse_attention_v3(q, k, v, ...)     # sparse_flash_attn_rf_v3.py
       ├─ do_tensor_rearrange_pooling(txt_len)  # 正确，支持 txt_len
       ├─ get_blockwise_mask_binary(txt_len)    # 正确，支持 txt_len
       ├─ rain_fusion_attention_v3(...)         # 正确，执行 BSA 算子
       └─ _bsa_inv_rearrange(out, tq, hq, wq)  # ← BUG：无 txt_len 参数
```

`_bsa_inv_rearrange` 的逻辑是：
```python
def _bsa_inv_rearrange(out, tq, hq, wq):
    first_frame_len = hq * wq
    out_first = out[:, :first_frame_len]         # 第一帧图像 token
    out_rest  = out[:, first_frame_len:]          # 剩余部分
    # 对 out_rest 执行逆重排：
    out_rest = out_rest.reshape(b, tq-1, hn, wn, 8, 8, n, d)  # ← 崩溃点
    ...
```

当 `tq=1`（单图）且 `txt_len>0` 时：
- `out_first` = 全部图像 token（`hq*wq` 个）
- `out_rest` = 文本 token（`txt_len` 个，**不为空**）
- 但函数以为 `out_rest` 是剩余图像帧 `(tq-1)*hq*wq = 0` 个
- 执行 `out_rest.reshape(b, 0, hn, wn, 8, 8, n, d)` → 期望 0 个元素，实际有 `txt_len` 个 → **reshape 报错**

本质原因：`_bsa_inv_rearrange` 在设计时假设序列中只有图像 token，不感知文本 token 的存在。

#### 为什么手动展开就可以了？

本地 MindIE-SD 源码（`sparse_flash_attn.py`）中的 rf_v3 路径实际上并不经过 `bsa_sparse_attention_v3`，而是直接调用底层函数并使用 `do_tensor_inv_rearrange`：

```
本地 MindIE-SD 的正确流程（sparse_flash_attn.py rf_v3 分支）：
  do_tensor_rearrange_pooling(txt_len)   # 重排图像 token，txt token 保持在末尾
  get_blockwise_mask(return_binary=True) # 生成 int8 binary mask
  rain_fusion_attention_v3(...)          # 执行 v3 BSA 算子
  do_tensor_inv_rearrange(txt_len)       # ← 关键：有 txt_len 参数
```

`do_tensor_inv_rearrange`（来自 rf_v2 模块）知道文本 token 的存在：
```python
def do_tensor_inv_rearrange(tensor, txt_len, latent_shape_q, ...):
    # 先分离图像部分和文本部分
    img_len = tq * hq * wq
    img_part = tensor[:, :img_len]   # 只对图像 token 做逆重排
    txt_part = tensor[:, img_len:]   # 文本 token 原样保留
    # 对 img_part 执行正确的逆重排
    img_part = _inv_rearrange_image_tokens(img_part, ...)
    return torch.cat([img_part, txt_part], dim=1)
```

因此，手动展开后：
1. `do_tensor_rearrange_pooling` 正确处理 txt_len ✓
2. `get_blockwise_mask(return_binary=True)` 生成 v3 所需的 int8 binary mask，且 text block 被强制保护 ✓
3. `rain_fusion_attention_v3` 执行的是完全相同的 NPU 算子 ✓
4. `do_tensor_inv_rearrange` 正确区分图像 token 和文本 token，各自处理 ✓

安装版和本地源码的差异本质是：安装版把 rf_v3 的高层封装（`bsa_sparse_attention_v3`）直接暴露出来，而该封装的逆重排函数是为纯图像序列（txt_len=0）设计的；本地源码的路由器则直接使用了更底层、支持 txt_len 的 `do_tensor_inv_rearrange`。手动展开等价于在应用层复现了本地源码的正确路径。

**正确做法：** 手动复现本地 MindIE-SD 源码中的 rf_v3 流程，使用 `do_tensor_inv_rearrange`（来自 rf_v2 模块，正确处理 txt_len）：

```python
from mindiesd.layers.flash_attn.sparse_flash_attn_rf_v2 import (
    do_tensor_rearrange_pooling, get_blockwise_mask, do_tensor_inv_rearrange
)
from mindiesd.layers.flash_attn.sparse_flash_attn_rf_v3 import rain_fusion_attention_v3

# 1. 空间重排 + avgpool
q_, k_, v_, qkv_pool = do_tensor_rearrange_pooling(
    sparse_q, sparse_k, sparse_v,
    txt_len, pool_size, latent_shape, latent_shape, "BSND"
)

# 2. 生成 int8 binary block_sparse_mask
block_sparse_mask = get_blockwise_mask(
    qkv_pool, txt_len, sparsity, scale,
    pool_size, latent_shape, latent_shape, "BSND",
    return_binary=True
)

# 3. S-padding（要求 S % pool_size == 0）
orig_seq_len = q_.shape[1]
pad_len = (-orig_seq_len) % pool_size
if pad_len > 0:
    q_ = F.pad(q_, (0, 0, 0, 0, 0, pad_len))
    k_ = F.pad(k_, (0, 0, 0, 0, 0, pad_len))
    v_ = F.pad(v_, (0, 0, 0, 0, 0, pad_len))
actual_seq_lens = [orig_seq_len] * q_.shape[0]

# 4. Block Sparse Attention（v3 算子）
out = rain_fusion_attention_v3(
    q_, k_, v_,
    block_sparse_mask=block_sparse_mask,
    scale=scale,
    head_num=head_num,
    input_layout="BSND",
    actual_seq_lengths=actual_seq_lens,
    actual_seq_lengths_kv=actual_seq_lens,
    sparse_size=pool_size,
    inner_precise=_BSA_INNER_PRECISE,
)

# 5. 裁掉 padding
if pad_len > 0:
    out = out[:, :orig_seq_len]

# 6. 逆重排（do_tensor_inv_rearrange 正确处理 txt_len）
out = do_tensor_inv_rearrange(out, txt_len, latent_shape, latent_shape, "BSND")
```

---

## 四、踩过的坑

| 问题 | 原因 | 解决 |
|------|------|------|
| `list index out of range` at `img_shapes[0][2]` | `img_shapes[0]` 是图像 shape 的列表，不是单个 shape tuple | 改为 `img_shapes[0][0][2]`，先取 batch 0，再取第一张图 |
| `img_shapes[0][1] // 2` 再次 `//2` | pipeline 构造 `img_shapes` 时已做 `// vae_scale_factor // 2`，不需要再除 | 直接用 `img_shapes[0][0][1]`，不做额外除法 |
| `8422 != 4096` einops 报错 | 传入了 joint 序列（txt+img=8422），但 `latent_shape=(1,64,64)` 只有 4096，`txt_len=0` 导致函数把所有 token 当图像处理 | 计算 `txt_len = total_seq - img_token_num`，并将序列重排为 `[img, txt]` |
| `reshape '[1, 0, 8, 8, 8, 8, 24, 128]' is invalid` | 安装版 `bsa_sparse_attention_v3` 的 `_bsa_inv_rearrange` 在 `tq=1, txt_len>0` 时，`out_rest` 含文本 token，但被 reshape 为 0 元素 | 绕过 `bsa_sparse_attention_v3`，直接调用底层 `rain_fusion_attention_v3` + `do_tensor_inv_rearrange` |

---

## 五、使用方式

```bash
python run_edit_2509.py \
  --model_path ./Qwen-Image-Edit-2509 \
  --use_rainfusion \
  --sparsity 0.64 \
  --sparse_start_step 15 \
  --img_paths ./test.png \
  --prompt_file ./prompts.txt \
  --width 1024 --height 1024 \
  --vae_tiling
```

- `--sparsity`：稀疏比例，越大速度越快但质量可能下降，默认 0.64
- `--sparse_start_step`：从第几步开始使用稀疏注意力（前几步高噪声，用 dense attention），默认 15
- 不加 `--use_rainfusion` 则完全走原始 dense attention 路径
