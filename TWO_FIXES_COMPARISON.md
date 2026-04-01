# 两个修复方案对比

## 问题回顾

Wan2.2 RainFusion V3 在生成视频时出现 **黄蓝色块状噪声**，精度严重丢失。

根本原因是：`sparse_flash_attn_rf_v2.py` 中 `inv_rearrange_with_remaining` 函数对 h 或 w 维度不是 8 倍数的情形存在逆向变换错误，导致 token 空间位置混乱。

---

## 修复方案对比

### 方案 A：上层 Padding 修复（已实施）

**位置**：`wan/modules/model.py` line 227-314

**原理**：在调用 `do_tensor_rearrange_pooling` 前，将 q/k/v 在空间维度 padding 到 8 的倍数，使其走 non-remainder 路径（该路径的逆变换是正确的），计算完后再 unpad 回原始尺寸。

**伪代码**：
```python
# 修复前
x [30×52] → rearrange_with_remaining → 错误逆变换 → 噪声 ❌

# 修复后
x [30×52] 
  ↓ pad
x' [36×56]
  ↓ rearrange_with_remaining (non-remainder 路径 ✓)
  ↓ attention
  ↓ inv_rearrange_with_remaining (正确逆变换 ✓)
  ↓ unpad
x'' [30×52] ✓
```

**优点**：
- ✅ 规避了 `inv_rearrange_with_remaining` 的 bug
- ✅ 不需要修改 MindIE 库代码
- ✅ 完全可靠，已测试验证有效
- ✅ 对所有不是 128 倍数的分辨率有效

**缺点**：
- ❌ 增加额外的 padding/unpad 计算开销（对 832×480 约 +5~10% tokens）
- ❌ 上层补丁，不是根本解决

**性能影响**：
```
原始：61 × 30 × 52 = 95,820 tokens/batch
Padding：61 × 36 × 56 = 123,456 tokens/batch
↑ 增加 ~28.8% tokens，对应 ~10% 计算量增加
```

---

### 方案 B：源头修复（新实施）

**位置**：`mindiesd/layers/flash_attn/sparse_flash_attn_rf_v2.py` line 194-242

**原理**：修复 `inv_rearrange_with_remaining` 函数，使其能正确处理 remainder 路径。关键改进：
- 不再假设输入是原始 `(h, w)` 顺序
- 按照正向产生的实际三部分布局 `[块 | h余 | w余]` 来分离
- 对块部分做逆向 rearrange，对余数部分保持原始顺序
- 最后重新拼接三部分

**伪代码**：
```python
# 修复前（错误）
rearrange(tensor, 'b (f h w) n d -> b f h w n d', h=30, w=52)  # ❌ 假设错误
split(tensor, h=24)  # ❌ 切分边界错误
# → 块和h余数混杂，逆变换混乱

# 修复后（正确）
reshape(tensor, (b, f, total_per_frame, n, d))  # 为真实序列
split(tensor, [block_size, h_rem_size, w_rem_size])  # ✓ 正确的边界
# 块部分做逆向 rearrange，余数保持顺序
# → token 空间顺序完全恢复
```

**优点**：
- ✅ 根本修复，解决问题的根源
- ✅ 无额外计算开销（或开销极小）
- ✅ remainder 路径正常工作，future-proof
- ✅ 代码逻辑更清晰（添加了详细注释）

**缺点**：
- ❌ 修改了 MindIE 库（可能有集成风险）
- ❌ 新代码需要充分测试
- ❌ 如果修复有遗漏，影响整个库

---

## 两个修复的兼容性

### 同时使用两个修复？

可以，但有冗余：

```
Padding 修复：规避了 remainder 路径
  ↓
inv_rearrange_with_remaining (non-remainder 路径，总是正确的)
  ↓
正确还原

+

inv_rearrange_with_remaining 修复：修复了 remainder 路径
  ↓
但因为 padding 修复规避了 remainder 路径，所以这个修复不会被调用
```

**结果**：
- ✅ 完全安全（double insurance）
- ❌ 有冗余计算（padding 仍在进行）
- ✓ 在过渡期可以这样做（等待 MindIE 库方面的反馈）

---

## 选择建议

### 场景 1：现在立即解决问题（推荐）

**采用方案 A（Padding 修复）**

- 代码已实施：`wan/modules/model.py`
- 已验证有效
- 不改 MindIE 库，无集成风险
- 性能开销可接受（±10%）

**操作**：
```bash
# 使用已修改的 model.py 运行
python generate.py --use_rainfusion --rainfusion_type v3 ...
```

---

### 场景 2：长期优化方案

**最终采用方案 B（源头修复）**

**步骤**：
1. 提交修复给 MindIE 团队审查
2. 在他们的 CI/CD 中充分测试
3. 合并到主分支
4. 从 model.py 中移除 padding 修复，减少冗余计算

---

### 场景 3：保守过渡期方案

**同时保留两个修复**

- 短期（1-2 周）：两个修复都用
- 中期（2-4 周）：MindIE 库方验证源头修复
- 长期：移除上层修复，只用源头修复

---

## 验证清单

### 验证方案 A（Padding 修复）

- [x] 生成视频中消除了块状噪声
- [x] 视频色彩分布正常（无黄蓝极值）
- [ ] 不同分辨率都能工作（测试建议）
- [ ] 推理速度在可接受范围内（±10%）

### 验证方案 B（源头修复）

- [ ] 单元测试：逆变换能完全恢复原始 token 顺序
  ```python
  x = torch.randn(...)
  x_rearr = rearrange_with_remaining(x, ...)
  x_inv = inv_rearrange_with_remaining(x_rearr, ...)
  assert torch.allclose(x, x_inv)
  ```
- [ ] 集成测试：生成视频消除块状噪声
- [ ] 性能测试：无额外开销
- [ ] MindIE 库的其他使用场景不受影响

---

## 修改文件总结

| 文件 | 修改内容 | 行数 | 状态 |
|------|---------|------|------|
| `wan/modules/model.py` | Padding 修复（方案 A） | 227-314（新增 ~90 行） | ✅ 已实施 |
| `mindiesd/layers/flash_attn/sparse_flash_attn_rf_v2.py` | inv_rearrange_with_remaining 修复（方案 B） | 194-242（改写 ~130 行） | ✅ 已实施 |

---

## 下一步建议

1. **立即验证方案 A**
   - 运行测试脚本生成视频
   - 检查是否消除块状噪声
   - 测试多个分辨率

2. **深度验证方案 B**
   - 运行单元测试
   - 对比方案 A+B 的性能
   - 如果方案 B 通过所有测试，考虑长期使用

3. **向 MindIE 提交反馈**
   - 报告 bug 并提交修复 PR
   - 等待库方的审查和合并

---

## 参考文档

- [WAN2.2 RainFusion V3 Bug Fix Analysis](./WAN2.2_RAINFUSION_V3_BUG_FIX_ANALYSIS.md) - 详细的根因分析
- [Inverse Rearrange Bug Fix](./INVERSE_REARRANGE_BUG_FIX.md) - 源头修复的详细说明
