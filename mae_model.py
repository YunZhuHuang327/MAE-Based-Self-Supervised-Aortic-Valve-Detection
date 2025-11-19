import torch
import torch.nn as nn
import numpy as np
from functools import partial
import timm


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding for MAE.
    Splits image into patches and projects them to embedding dimension.
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_chans, embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)  # (B, embed_dim, H/patch_size, W/patch_size)
        x = x.flatten(2)   # (B, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


class MAEEncoder(nn.Module):
    """
    Vision Transformer encoder for MAE.
    Only processes visible (unmasked) patches.
    """

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm
    ):
        super().__init__()

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim),
            requires_grad=False
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            timm.models.vision_transformer.Block(
                embed_dim,
                num_heads,
                mlp_ratio,
                qkv_bias=True,
                norm_layer=norm_layer
            )
            for _ in range(depth)
        ])

        self.norm = norm_layer(embed_dim)

        # Initialize positional embeddings
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize positional embeddings with sin-cos
        pos_embed = self.get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.patch_embed.num_patches ** 0.5),
            cls_token=True
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize cls token
        nn.init.normal_(self.cls_token, std=0.02)

    @staticmethod
    def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
        """Generate 2D sin-cos positional embeddings."""
        grid_h = np.arange(grid_size, dtype=np.float32)
        grid_w = np.arange(grid_size, dtype=np.float32)
        grid = np.meshgrid(grid_w, grid_h)
        grid = np.stack(grid, axis=0)  # (2, grid_size, grid_size)

        grid = grid.reshape([2, 1, grid_size, grid_size])
        pos_embed = MAEEncoder.get_2d_sincos_pos_embed_from_grid(embed_dim, grid)

        if cls_token:
            pos_embed = np.concatenate(
                [np.zeros([1, embed_dim]), pos_embed], axis=0
            )
        return pos_embed

    @staticmethod
    def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
        assert embed_dim % 2 == 0

        # Use half of dimensions to encode grid_h, other half for grid_w
        emb_h = MAEEncoder.get_1d_sincos_pos_embed_from_grid(
            embed_dim // 2, grid[0]
        )
        emb_w = MAEEncoder.get_1d_sincos_pos_embed_from_grid(
            embed_dim // 2, grid[1]
        )

        emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, embed_dim)
        return emb

    @staticmethod
    def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
        """Generate 1D sin-cos positional embeddings."""
        omega = np.arange(embed_dim // 2, dtype=np.float32)
        omega /= embed_dim / 2.0
        omega = 1.0 / 10000 ** omega  # (embed_dim/2,)

        pos = pos.reshape(-1)  # (M,)
        out = np.einsum("m,d->md", pos, omega)  # (M, embed_dim/2)

        emb_sin = np.sin(out)
        emb_cos = np.cos(out)

        emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, embed_dim)
        return emb

    def forward(self, x, mask):
        """
        Forward pass through encoder.

        Args:
            x: Input images (B, C, H, W)
            mask: Binary mask (B, num_patches) - 1 means keep, 0 means remove

        Returns:
            Encoded visible patches
        """
        # Patch embedding
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)

        # Add positional embeddings (excluding cls token)
        x = x + self.pos_embed[:, 1:, :]

        # Apply mask - keep only visible patches
        B, N, D = x.shape
        mask = mask.unsqueeze(-1).expand_as(x)  # (B, N, D)
        x = x[mask.bool()].reshape(B, -1, D)  # (B, num_visible, embed_dim)

        # Add cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        return x


class MAEDecoder(nn.Module):
    """
    Lightweight decoder for MAE.
    Reconstructs original image from encoded patches.
    """

    def __init__(
        self,
        num_patches,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm
    ):
        super().__init__()

        self.num_patches = num_patches
        self.patch_size = patch_size
        self.in_chans = in_chans

        # Project from encoder to decoder dimension
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        # Mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        # Decoder positional embeddings
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_embed_dim),
            requires_grad=False
        )

        # Transformer blocks
        self.decoder_blocks = nn.ModuleList([
            timm.models.vision_transformer.Block(
                decoder_embed_dim,
                decoder_num_heads,
                mlp_ratio,
                qkv_bias=True,
                norm_layer=norm_layer
            )
            for _ in range(decoder_depth)
        ])

        self.decoder_norm = norm_layer(decoder_embed_dim)

        # Reconstruct pixels
        self.decoder_pred = nn.Linear(
            decoder_embed_dim,
            patch_size ** 2 * in_chans,
            bias=True
        )

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize positional embeddings
        decoder_pos_embed = MAEEncoder.get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1],
            int(self.num_patches ** 0.5),
            cls_token=True
        )
        self.decoder_pos_embed.data.copy_(
            torch.from_numpy(decoder_pos_embed).float().unsqueeze(0)
        )

        # Initialize mask token
        nn.init.normal_(self.mask_token, std=0.02)

    def forward(self, x, ids_restore):
        """
        Forward pass through decoder.

        Args:
            x: Encoded visible patches (B, num_visible+1, embed_dim)
            ids_restore: Indices to restore original order (B, num_patches)

        Returns:
            Reconstructed patches (B, num_patches, patch_size^2 * in_chans)
        """
        # Project to decoder dimension
        x = self.decoder_embed(x)

        # Separate cls token and visible patches
        cls_token = x[:, :1, :]
        x = x[:, 1:, :]

        # Append mask tokens
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] - x.shape[1], 1
        )
        x = torch.cat([x, mask_tokens], dim=1)  # (B, num_patches, decoder_embed_dim)

        # Restore original order
        x = torch.gather(
            x, dim=1,
            index=ids_restore.unsqueeze(-1).expand(-1, -1, x.shape[2])
        )

        # Add cls token back
        x = torch.cat([cls_token, x], dim=1)

        # Add positional embeddings
        x = x + self.decoder_pos_embed

        # Apply transformer blocks
        for block in self.decoder_blocks:
            x = block(x)

        x = self.decoder_norm(x)

        # Remove cls token
        x = x[:, 1:, :]

        # Reconstruct pixels
        x = self.decoder_pred(x)

        return x


class MAE(nn.Module):
    """
    Masked Autoencoder (MAE) for medical images.

    Reference:
    He et al. "Masked Autoencoders Are Scalable Vision Learners"
    https://arxiv.org/abs/2111.06377
    """

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.0,
        mask_ratio=0.75,
        norm_layer=nn.LayerNorm
    ):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.num_patches = (img_size // patch_size) ** 2

        # Encoder
        self.encoder = MAEEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            norm_layer=norm_layer
        )

        # Decoder
        self.decoder = MAEDecoder(
            num_patches=self.num_patches,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            decoder_embed_dim=decoder_embed_dim,
            decoder_depth=decoder_depth,
            decoder_num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio,
            norm_layer=norm_layer
        )

    def random_masking(self, x, mask_ratio):
        """
        Perform random masking by shuffling patches.

        Args:
            x: Patch embeddings (B, N, D)
            mask_ratio: Ratio of patches to mask

        Returns:
            mask: Binary mask (B, N) - 1 is keep, 0 is remove
            ids_restore: Indices to restore original order
        """
        B, N, D = x.shape
        len_keep = int(N * (1 - mask_ratio))

        # Random shuffle
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # Create mask: 1 is keep, 0 is remove
        mask = torch.zeros([B, N], device=x.device)
        mask[:, :len_keep] = 1

        # Unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return mask, ids_restore

    def patchify(self, imgs):
        """
        Convert images to patches.

        Args:
            imgs: (B, C, H, W)

        Returns:
            patches: (B, num_patches, patch_size^2 * C)
        """
        p = self.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        c = imgs.shape[1]
        x = imgs.reshape(imgs.shape[0], c, h, p, w, p)
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(imgs.shape[0], h * w, p ** 2 * c)
        return x

    def unpatchify(self, x):
        """
        Convert patches back to images.

        Args:
            x: (B, num_patches, patch_size^2 * C)

        Returns:
            imgs: (B, C, H, W)
        """
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        c = 3  # RGB channels
        x = x.reshape(x.shape[0], h, w, p, p, c)
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(x.shape[0], c, h * p, w * p)
        return imgs

    def forward_encoder(self, x, mask_ratio=None):
        """Encode images with random masking."""
        if mask_ratio is None:
            mask_ratio = self.mask_ratio

        # Create patches
        x = self.encoder.patch_embed(x)  # (B, num_patches, embed_dim)

        # Add positional embeddings (without cls token)
        x = x + self.encoder.pos_embed[:, 1:, :]

        # Generate random mask
        mask, ids_restore = self.random_masking(x, mask_ratio)

        # Apply mask - keep only visible patches
        B, N, D = x.shape
        mask_expanded = mask.unsqueeze(-1).expand_as(x)  # (B, N, D)
        x_visible = x[mask_expanded.bool()].reshape(B, -1, D)  # (B, num_visible, D)

        # Add cls token
        cls_token = self.encoder.cls_token + self.encoder.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x_visible.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x_visible), dim=1)  # (B, num_visible+1, D)

        # Apply transformer blocks
        for block in self.encoder.blocks:
            x = block(x)

        x = self.encoder.norm(x)

        return x, mask, ids_restore

    def forward_loss(self, imgs, pred, mask):
        """
        Compute reconstruction loss.

        Args:
            imgs: Original images (B, C, H, W)
            pred: Predicted patches (B, num_patches, patch_size^2 * C)
            mask: Binary mask (B, num_patches) - 1 is remove, 0 is keep

        Returns:
            loss: MSE loss on masked patches only
        """
        target = self.patchify(imgs)

        # Compute loss only on masked patches
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # Mean per patch

        # Invert mask for loss computation (1 is remove, 0 is keep)
        mask = 1 - mask

        loss = (loss * mask).sum() / mask.sum()  # Mean loss on removed patches
        return loss

    def forward(self, imgs, mask_ratio=None):
        """
        Forward pass through MAE.

        Args:
            imgs: Input images (B, C, H, W)
            mask_ratio: Ratio of patches to mask (overrides default)

        Returns:
            loss: Reconstruction loss
            pred: Predicted image
            mask: Binary mask
        """
        # Encode with masking
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)

        # Decode
        pred = self.decoder(latent, ids_restore)

        # Compute loss
        loss = self.forward_loss(imgs, pred, mask)

        return loss, pred, mask


def build_mae_model(
    img_size=224,
    patch_size=16,
    in_chans=3,
    embed_dim=768,
    depth=12,
    num_heads=12,
    decoder_embed_dim=512,
    decoder_depth=8,
    decoder_num_heads=16,
    mask_ratio=0.75
):
    """
    Factory function to create MAE model.

    Args:
        img_size: Input image size
        patch_size: Size of each patch
        in_chans: Number of input channels
        embed_dim: Encoder embedding dimension
        depth: Encoder depth
        num_heads: Encoder number of attention heads
        decoder_embed_dim: Decoder embedding dimension
        decoder_depth: Decoder depth
        decoder_num_heads: Decoder number of attention heads
        mask_ratio: Ratio of patches to mask

    Returns:
        MAE model
    """
    model = MAE(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        decoder_embed_dim=decoder_embed_dim,
        decoder_depth=decoder_depth,
        decoder_num_heads=decoder_num_heads,
        mlp_ratio=4.0,
        mask_ratio=mask_ratio,
        norm_layer=partial(nn.LayerNorm, eps=1e-6)
    )

    return model
