# Bug 修正說明

## 問題描述

訓練時出現錯誤:
```
RuntimeError: shape '[128, 768, 14, 16, 14, 16]' is invalid for input of size 19267584
```

## 問題原因

1. **`forward_encoder` 方法錯誤**: 原本的實作嘗試對已經轉換的 patch embeddings 進行 patchify/unpatchify 操作，這是不正確的
2. **`patchify` 方法錯誤**: 使用了錯誤的通道數推斷
3. **棄用警告**: 使用了舊版的混合精度 API

## 修正內容

### 1. 修正 `forward_encoder` 方法 (mae_model.py:422-452)

**修正前**:
```python
def forward_encoder(self, x, mask_ratio=None):
    # ...
    x = self.encoder.patch_embed(x)
    mask, ids_restore = self.random_masking(x, mask_ratio)
    # 錯誤: 對 embeddings 進行 patchify/unpatchify
    x = self.encoder(self.unpatchify(self.patchify(...)), mask)
    return x, mask, ids_restore
```

**修正後**:
```python
def forward_encoder(self, x, mask_ratio=None):
    # 1. Patch embedding
    x = self.encoder.patch_embed(x)  # (B, num_patches, embed_dim)

    # 2. Add positional embeddings
    x = x + self.encoder.pos_embed[:, 1:, :]

    # 3. Generate mask
    mask, ids_restore = self.random_masking(x, mask_ratio)

    # 4. Apply mask - keep only visible patches
    B, N, D = x.shape
    mask_expanded = mask.unsqueeze(-1).expand_as(x)
    x_visible = x[mask_expanded.bool()].reshape(B, -1, D)

    # 5. Add cls token
    cls_token = self.encoder.cls_token + self.encoder.pos_embed[:, :1, :]
    cls_tokens = cls_token.expand(x_visible.shape[0], -1, -1)
    x = torch.cat((cls_tokens, x_visible), dim=1)

    # 6. Apply transformer
    for block in self.encoder.blocks:
        x = block(x)
    x = self.encoder.norm(x)

    return x, mask, ids_restore
```

### 2. 修正 `patchify` 和 `unpatchify` 方法 (mae_model.py:384-422)

**修正前**:
```python
def patchify(self, imgs):
    # 錯誤: 使用 self.encoder.patch_embed.proj.out_channels
    x = imgs.reshape(imgs.shape[0], self.encoder.patch_embed.proj.out_channels, h, p, w, p)
    # ...
```

**修正後**:
```python
def patchify(self, imgs):
    # 正確: 使用實際的圖像通道數
    c = imgs.shape[1]  # RGB = 3
    x = imgs.reshape(imgs.shape[0], c, h, p, w, p)
    x = torch.einsum('nchpwq->nhwpqc', x)
    x = x.reshape(imgs.shape[0], h * w, p ** 2 * c)
    return x
```

### 3. 更新混合精度 API (mae_train.py:110, 236)

**修正前**:
```python
with torch.cuda.amp.autocast():
    loss, _, _ = model(images)

scaler = torch.cuda.amp.GradScaler()
```

**修正後**:
```python
with torch.amp.autocast('cuda'):
    loss, _, _ = model(images)

scaler = torch.amp.GradScaler('cuda')
```

## 測試

修正後應該能正常訓練，可以重新執行:

```bash
./run_training_valve_optimized.sh
```

## 影響範圍

- ✅ `mae_model.py` - 核心模型邏輯
- ✅ `mae_train.py` - 訓練腳本
- ⚠️ 不影響其他文件（dataset, evaluate 等）

## 驗證

訓練正常啟動後，應該看到:
1. Loss 開始下降
2. 每個 iteration 正常完成
3. 無 shape mismatch 錯誤

---

修正完成日期: 2025-11-03
