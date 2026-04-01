# sparse_flash_attn_rf_v2.py 中 inv_rearrange_with_remaining 的 Bug 修复

## 问题描述

`inv_rearrange_with_remaining` 函数在处理 h 或 w 维度不是 8 倍数的情形时，无法正确还原 token 的空间顺序，导致块状噪声。

---

## Bug 根源

### 正向 rearrange_with_remaining 的产出

正向重排对每帧产生了三部分的序列布局：

```
[块重排部分(1152 tokens) | h余数部分(312 tokens) | w余数部分(96 tokens)]
```

具体来说：
- **块重排**：`(hq // 8) * (wq // 8) * 64` tokens
- **h余数**：`(hq % 8) * wq` tokens
- **w余数**：`(hq // 8 * 8) * (wq % 8)` tokens

对于 hq=30, wq=52 的例子：
- 块重排：3 * 6 * 64 = 1152 tokens
- h余数：6 * 52 = 312 tokens
- w余数：24 * 4 = 96 tokens
- **总计**：1560 tokens/frame

### ❌ 原代码的错误假设

```python
# 错误！假设输入是原始 (h=30, w=52) 顺序
tensor_hwt = rearrange(tensor, 'b (f h w) n d -> b f h w n d', 
                       f=frame_num - 1, h=hq, w=wq)
```

然后错误地按 h=24 切分：
```python
tensor_hwt, tensor_h_r = torch.split(tensor_hwt, hq - (hq % 8), dim=2)
# h=24 的切分边界根本对不上 1152 的块大小边界！
```

**结果**：h余数和块重排的部分被混杂在一起，逆变换完全混乱。

---

## ✓ 修复方案

### 核心思路

不再假设输入是原始 `(h, w)` 顺序，而是**按照正向产生的实际布局** `[块 | h余 | w余]` 来分离和恢复：

```python
# Step 1: 计算三部分的实际大小
block_size = (hq // 8) * (wq // 8) * 64
h_rem_size = (hq % 8) * wq
w_rem_size = (hq // 8 * 8) * (wq % 8)
total_per_frame = block_size + h_rem_size + w_rem_size

# Step 2: 按这些大小从序列中分离三部分
tensor = tensor.reshape(b, frame_num - 1, total_per_frame, n, d)
tensor_hwt = tensor[:, :, :block_size, :, :]              # 块重排部分
tensor_h_r = tensor[:, :, block_size:block_size + h_rem_size, :, :]  # h余数
tensor_w_r = tensor[:, :, block_size + h_rem_size:, :, :]  # w余数

# Step 3: 对块部分做逆向 rearrange，余数部分保持原始顺序
tensor_hwt = rearrange(tensor_hwt, 'b f (hn wn hb wb) n d -> b f (hn hb wn wb) n d',
                      hb=8, wb=8, hn=hq // 8, wn=wq // 8)

# Step 4: 恢复到空间形状并重组三部分
tensor_hwt = tensor_hwt.reshape(b, frame_num - 1, hq_block, wq_block, n, d)
tensor_h_r = tensor_h_r.reshape(b, frame_num - 1, hq_rem, wq, n, d)
tensor_w_r = tensor_w_r.reshape(b, frame_num - 1, hq_block, wq_rem, n, d)

# 拼接：[块区域(hq_block × wq_block)] + [h余(hq_rem × wq)] → 高度方向
#       然后 + [w余(hq × wq_rem)] → 宽度方向
tensor_hwt = torch.cat([tensor_hwt, tensor_h_r], dim=2)  # 沿 h 方向
tensor_hwt = torch.cat([tensor_hwt, tensor_w_r], dim=3)  # 沿 w 方向
```

---

## 修复的实现细节

### 处理边界情况

修复代码支持以下组合：
- `hq % 8 = 0, wq % 8 ≠ 0`：只有 w 余数
- `hq % 8 ≠ 0, wq % 8 = 0`：只有 h 余数
- `hq % 8 ≠ 0, wq % 8 ≠ 0`：同时有 h 和 w 余数

通过条件判断 `if block_size > 0`, `if h_rem_size > 0`, `if w_rem_size > 0` 分别处理。

### 两种 layout 的支持

修复同时处理 **BSND** 和 **BNSD** 两种布局：

**BSND** (`[B, S, N, D]`)：
```python
tensor.reshape(b, frame_num - 1, total_per_frame, n, d)
# 沿维度 2 (seq_len) 分离三部分
```

**BNSD** (`[B, N, S, D]`)：
```python
tensor.reshape(b, n, frame_num - 1, total_per_frame, d)
# 沿维度 3 (seq_len) 分离三部分
```

---

## 验证正确性

修复后的 `inv_rearrange_with_remaining` 应该满足以下性质：

```python
# 正向再逆向，应该恢复原始
x_orig = torch.randn(...)  # 原始空间顺序
x_rearr = rearrange_with_remaining(x_orig, ...)
x_inv = inv_rearrange_with_remaining(x_rearr, ...)
assert torch.allclose(x_orig, x_inv, atol=1e-5)  # 应该完全恢复
```

---

## 与 model.py 的上层 Padding 修复的关系

### 两种修复方案

| 修复位置 | 方案 | 优点 | 缺点 |
|---------|------|------|------|
| **model.py**（上层） | Padding 到 8 倍数，走 non-remainder 路径 | 规避 bug，完全安全；不改 MindIE 代码 | 多余的 padding 计算 |
| **sparse_flash_attn_rf_v2.py**（源头） | 修复 remainder 路径逻辑 | 根本修复；无额外计算 | 依赖修复代码的正确性 |

### 推荐做法

**选项 1（保守，推荐现在使用）**：
- 保留 model.py 中的 padding 修复
- 禁用 sparse_flash_attn_rf_v2.py 中的修复（暂不使用）
- 原因：padding 修复已经彻底规避了 bug，改 MindIE 代码有风险

**选项 2（激进，长期方案）**：
- 移除 model.py 中的 padding 修复
- 使用 sparse_flash_attn_rf_v2.py 中的修复
- 原因：更干净，无额外计算，但需要充分测试

**选项 3（过渡方案）**：
- 同时使用两个修复（提交给 MindIE 等待审查期间的安全做法）
- 虽然 padding 此时变得冗余，但不影响正确性

---

## 文件修改记录

### sparse_flash_attn_rf_v2.py

**修改函数**：`inv_rearrange_with_remaining` (line 194-242)

**主要改动**：
1. 删除错误的 `rearrange(..., f=frame_num-1, h=hq, w=wq)` 假设
2. 添加三部分大小计算
3. 用 reshape + split 按正确边界分离三部分
4. 对块部分做逆向 rearrange，对余数部分保持顺序
5. 用 torch.cat 重组三部分回空间顺序

**行数**：~130 行（从原来的 ~50 行，因为添加了详细注释和完整的边界处理）

---

## 测试建议

### 单元测试

```python
def test_inv_rearrange_correctness():
    """验证逆变换的正确性"""
    for hq in [30, 25, 31, 16, 24]:  # 各种不是 8 倍数的情况
        for wq in [52, 50, 55, 32, 48]:
            x = torch.randn(2, 61*hq*wq, 16, 128)  # [B, S, N, D]
            latent_shape = (61, hq, wq)
            
            x_rearr = rearrange_with_remaining(x, latent_shape, latent_shape, "BSND")
            x_inv = inv_rearrange_with_remaining(x_rearr, latent_shape, latent_shape, "BSND")
            
            assert torch.allclose(x, x_inv, atol=1e-5), f"Failed for hq={hq}, wq={wq}"
```

### 集成测试

```bash
# 用修复后的库生成视频，对比输出质量
python generate.py --use_rainfusion --rainfusion_type v3 \
    --image examples/i2v_input.JPG \
    --prompt "..."
# 检查是否消除了块状噪声
```

---

## 备注

- 这个修复不改变 API，完全向后兼容
- 修复同时改进了代码可读性（详细的注释说明了三部分的含义）
- 如果发现新的边界情况，可以在 torch.cat 的逻辑中补充

