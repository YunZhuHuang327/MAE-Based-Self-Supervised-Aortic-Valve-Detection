"""
MAE + YOLO Integration for Heart Valve Detection

This module integrates the pre-trained MAE encoder as a backbone
for YOLO object detection model.
"""

import torch
import torch.nn as nn
from mae_model import build_mae_model, MAEEncoder
from ultralytics import YOLO
from ultralytics.nn.modules import Conv, C2f, SPPF, Detect
from ultralytics.nn.tasks import DetectionModel


class MAEBackbone(nn.Module):
    """
    MAE Encoder as a feature extraction backbone for YOLO.

    Extracts multi-scale features from MAE encoder for YOLO neck.
    """

    def __init__(
        self,
        mae_checkpoint,
        img_size=640,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        freeze_encoder=False,
        feature_layers=[3, 7, 11]  # Which transformer blocks to extract features from
    ):
        """
        Args:
            mae_checkpoint: Path to pre-trained MAE model
            img_size: Input image size
            patch_size: Patch size
            embed_dim: Embedding dimension
            depth: Number of transformer blocks
            num_heads: Number of attention heads
            freeze_encoder: Whether to freeze MAE encoder weights
            feature_layers: Which transformer layers to extract features from
        """
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.feature_layers = feature_layers

        # Load pre-trained MAE model
        print(f"Loading pre-trained MAE from {mae_checkpoint}")
        mae_model = build_mae_model(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=3,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads
        )

        checkpoint = torch.load(mae_checkpoint, map_location='cpu')
        mae_model.load_state_dict(checkpoint['model_state_dict'])

        # Extract encoder
        self.encoder = mae_model.encoder

        # Freeze encoder if specified
        if freeze_encoder:
            print("Freezing MAE encoder weights")
            for param in self.encoder.parameters():
                param.requires_grad = False
        else:
            print("MAE encoder will be fine-tuned")

        # Calculate grid size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2

        # Feature projection layers to convert to CNN-style features
        # YOLO expects features with shape (B, C, H, W)
        self.feature_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 256)
            ) for _ in feature_layers
        ])

        # Define output channels for each feature level (for YOLO neck)
        self.out_channels = [256, 256, 256]  # P3, P4, P5 equivalent

    def forward(self, x):
        """
        Forward pass to extract multi-scale features.

        Args:
            x: Input images (B, 3, H, W)

        Returns:
            List of feature maps at different scales
        """
        B = x.shape[0]

        # Patch embedding
        x = self.encoder.patch_embed(x)  # (B, num_patches, embed_dim)

        # Add positional embeddings
        x = x + self.encoder.pos_embed[:, 1:, :]

        # Add cls token
        cls_token = self.encoder.cls_token + self.encoder.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, num_patches+1, embed_dim)

        # Extract features from different layers
        features = []
        for i, block in enumerate(self.encoder.blocks):
            x = block(x)

            if i in self.feature_layers:
                # Remove cls token
                patch_features = x[:, 1:, :]  # (B, num_patches, embed_dim)

                # Project features
                proj_idx = self.feature_layers.index(i)
                proj_features = self.feature_projections[proj_idx](patch_features)

                # Reshape to CNN format (B, C, H, W)
                feat_map = proj_features.transpose(1, 2).reshape(
                    B, 256, self.grid_size, self.grid_size
                )
                features.append(feat_map)

        x = self.encoder.norm(x)

        return features


class MAEYOLOModel(nn.Module):
    """
    Complete YOLO model with MAE backbone.
    """

    def __init__(
        self,
        mae_checkpoint,
        num_classes=1,
        img_size=640,
        freeze_backbone=False
    ):
        super().__init__()

        self.num_classes = num_classes
        self.img_size = img_size

        # MAE backbone
        self.backbone = MAEBackbone(
            mae_checkpoint=mae_checkpoint,
            img_size=img_size,
            freeze_encoder=freeze_backbone
        )

        # YOLO neck (FPN-style)
        self.neck = nn.ModuleList([
            # Upsample and concatenate features
            nn.Sequential(
                Conv(256, 256, 1, 1),  # C3
                nn.Upsample(scale_factor=2, mode='nearest')
            ),
            nn.Sequential(
                Conv(512, 256, 1, 1),  # After concat
                C2f(256, 256, n=3)
            ),
            nn.Sequential(
                Conv(256, 256, 1, 1),  # C4
                nn.Upsample(scale_factor=2, mode='nearest')
            ),
            nn.Sequential(
                Conv(512, 128, 1, 1),  # After concat
                C2f(128, 128, n=3)
            ),
        ])

        # YOLO head
        self.detect = Detect(num_classes, [128, 256, 256])

    def forward(self, x):
        # Extract features from MAE backbone
        p3, p4, p5 = self.backbone(x)

        # YOLO neck (top-down)
        # ... (simplified, actual implementation depends on YOLO version)

        # Detection head
        outputs = self.detect([p3, p4, p5])

        return outputs


def create_mae_yolo_model(
    mae_checkpoint,
    yolo_config='yolov8s.yaml',
    num_classes=1,
    img_size=640,
    freeze_backbone=False
):
    """
    Factory function to create MAE+YOLO model.

    Args:
        mae_checkpoint: Path to pre-trained MAE checkpoint
        yolo_config: YOLO configuration
        num_classes: Number of detection classes
        img_size: Input image size
        freeze_backbone: Whether to freeze MAE backbone

    Returns:
        Integrated model
    """
    # Load baseline YOLO model
    base_model = YOLO(yolo_config)

    # Replace backbone with MAE
    model = MAEYOLOModel(
        mae_checkpoint=mae_checkpoint,
        num_classes=num_classes,
        img_size=img_size,
        freeze_backbone=freeze_backbone
    )

    print(f"Created MAE+YOLO model:")
    print(f"  - MAE checkpoint: {mae_checkpoint}")
    print(f"  - Backbone frozen: {freeze_backbone}")
    print(f"  - Image size: {img_size}")
    print(f"  - Number of classes: {num_classes}")

    return model


def load_mae_for_yolo(mae_checkpoint, freeze=True):
    """
    Load pre-trained MAE encoder for YOLO integration.

    Args:
        mae_checkpoint: Path to MAE checkpoint
        freeze: Whether to freeze weights

    Returns:
        MAE backbone module
    """
    backbone = MAEBackbone(
        mae_checkpoint=mae_checkpoint,
        freeze_encoder=freeze
    )

    # Count parameters
    total_params = sum(p.numel() for p in backbone.parameters())
    trainable_params = sum(p.numel() for p in backbone.parameters() if p.requires_grad)

    print(f"\nMAE Backbone Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Frozen parameters: {total_params - trainable_params:,}")

    return backbone


if __name__ == '__main__':
    # Test the model
    print("Testing MAE+YOLO integration...")

    # Create dummy checkpoint for testing
    mae_checkpoint = './mae_valve_output/best_model.pth'

    if not os.path.exists(mae_checkpoint):
        print(f"MAE checkpoint not found: {mae_checkpoint}")
        print("Please train MAE first using: ./run_training_valve_optimized.sh")
    else:
        # Load MAE backbone
        backbone = load_mae_for_yolo(mae_checkpoint, freeze=True)

        # Test forward pass
        dummy_input = torch.randn(2, 3, 640, 640)
        features = backbone(dummy_input)

        print(f"\nOutput features:")
        for i, feat in enumerate(features):
            print(f"  Feature {i}: {feat.shape}")
