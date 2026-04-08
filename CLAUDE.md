# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Purpose

This is a **documentation and analysis repository** tracking bug fixes and model adaptations for Block Sparse Attention (BSA) on Huawei Ascend NPU via the MindIE-SD library. There is no runnable code here — the actual code lives in separate model repos (Wan2.2, Qwen-Image-Edit-2509) and the MindIE-SD library.

## Committing

**Always invoke the `gitmoji_commit` skill before committing.** Use `/gitmoji_commit` to generate a Gitmoji-compliant commit message and execute the commit.

## Document Map

| File | Covers |
|------|--------|
| `WAN2.2_RAINFUSION_V3_BUG_FIX_ANALYSIS.md` | Root-cause analysis of the `inv_rearrange_with_remaining` bug (h/w not multiples of 8 trigger incorrect remainder path) |
| `INVERSE_REARRANGE_BUG_FIX.md` | Source-level fix for `inv_rearrange_with_remaining` in `sparse_flash_attn_rf_v2.py` (Method B) |
| `TWO_FIXES_COMPARISON.md` | Comparison of Method A (spatial padding in `model.py`) vs Method B (library source fix) |
| `rainfusion_v3_adaptation.md` | Step-by-step adaptation log for connecting Qwen-Image-Edit-2509 to BSA v3 |
| `bsa_v3_qwen_adaptation.md` | Final reference: all key pitfalls and the correct implementation for Qwen joint attention + BSA |

## Key Architecture Concepts

### Affected Codebases (external to this repo)

- **`wan/modules/model.py`** — Wan2.2 attention module; contains Method A (padding fix, lines 227–314)
- **`mindiesd/layers/flash_attn/sparse_flash_attn_rf_v2.py`** — MindIE library; contains Method B (`inv_rearrange_with_remaining` fix, lines 194–242). Changes here must be synced to the installed site-packages path on the target machine.
- **`mindiesd/layers/flash_attn/sparse_flash_attn_rf_v3.py`** — MindIE library; `bsa_sparse_attention_v3` wrapper. Needs patch to branch on `txt_len > 0` and call `do_tensor_inv_rearrange` instead of `_bsa_inv_rearrange`.
- **`Qwen-Image-Edit-2509/qwenimage_edit/transformer_qwenimage.py`** — Core Qwen transformer; `QwenDoubleStreamAttnProcessor2_0` is where BSA is integrated.
- **`Qwen-Image-Edit-2509/qwenimage_edit/pipeline_qwenimage_edit_plus.py`** — Denoising loop; passes `t_idx=i` to the transformer.

### The `inv_rearrange_with_remaining` Bug (Wan2.2)

`rearrange_with_remaining` produces a token layout of `[block_rearranged | h_remainder | w_remainder]` per frame. The old inverse function incorrectly assumed tokens were still in original `(h, w)` row-major order and split at the wrong boundary — mixing block tokens with h-remainder tokens.

- **Method A** (recommended for immediate use): pad `q/k/v` to the next multiple of 8 in h and w before calling BSA, then unpad. This forces the non-remainder path, which has a correct symbolic inverse.
- **Method B** (long-term): fix the remainder-path logic in the library itself; split by the actual `[block_size, h_rem_size, w_rem_size]` boundaries.

Both methods are safe to use simultaneously during a transition period.

### Qwen Joint Attention Differences from Wan2.2

| Aspect | Wan2.2 | Qwen-Image-Edit-2509 |
|--------|--------|----------------------|
| Token order | `[img, txt]` | `[txt, img]` |
| Attention structure | Separate image/text attention | Joint attention (concatenated) |
| `latent_shape` source | `grid_sizes` directly | Inferred from `img_shapes[0][0]` + actual token count |
| `txt_len` in BSA | 0 | `txt_query.shape[1]` |

**Critical**: `bsa_sparse_attention_v3` expects `[txt, img]` order (txt protected at the front). Qwen already uses `[txt, img]` — do **not** reorder. Early code incorrectly swapped to `[img, txt]` before calling BSA, causing semantic errors.

### `t_idx` vs Block Index

The `rainfusion_config["skip_timesteps"]` and `mask_refresh_interval` logic both require the **denoising step index** (from the outer loop `for i, t in enumerate(timesteps)`). A prior bug passed the **transformer block index** (`index_block`) instead, breaking both the skip-timesteps gate and the mask cache.

### Mask Caching (`mask_refresh_interval`)

Stored as `self._bsa_mask_cache` on the processor. `0` means reuse indefinitely. Cache is cleared when `t_idx < skip_timesteps`. Skipping mask recomputation saves ~863µs/step per block.

### Performance Note

BSA underperforms standard FlashAttention for sequences shorter than ~16K tokens. For 1024×1024 single-image editing (~8422 total tokens), BSA adds ~25% latency. BSA is intended for video or multi-image scenarios with much longer sequences.
