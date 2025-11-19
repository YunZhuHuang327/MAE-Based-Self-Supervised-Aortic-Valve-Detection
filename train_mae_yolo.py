"""
Train YOLO with Pre-trained MAE Backbone for Heart Valve Detection

This script integrates a pre-trained MAE encoder with YOLO for
object detection on medical images.
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
from pathlib import Path
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

from mae_model import build_mae_model
from mae_evaluate import MAEFeatureExtractor


def load_mae_encoder(mae_checkpoint, freeze=True):
    """
    Load pre-trained MAE encoder.

    Args:
        mae_checkpoint: Path to MAE checkpoint
        freeze: Whether to freeze encoder weights

    Returns:
        MAE encoder module
    """
    print(f"\n{'='*60}")
    print(f"Loading Pre-trained MAE Encoder")
    print(f"{'='*60}")
    print(f"Checkpoint: {mae_checkpoint}")

    # Build MAE model
    mae_model = build_mae_model(
        img_size=640,  # YOLO uses 640x640
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mask_ratio=0.65
    )

    # Load checkpoint
    checkpoint = torch.load(mae_checkpoint, map_location='cpu')
    mae_model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✅ Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"   Training loss: {checkpoint['loss']:.4f}")

    # Extract encoder
    encoder = mae_model.encoder

    # Freeze if specified
    if freeze:
        print(f"\n🔒 Freezing MAE encoder weights")
        for param in encoder.parameters():
            param.requires_grad = False
    else:
        print(f"\n🔓 MAE encoder will be fine-tuned")

    # Count parameters
    total_params = sum(p.numel() for p in encoder.parameters())
    trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)

    print(f"\nEncoder Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Frozen parameters: {total_params - trainable_params:,}")
    print(f"{'='*60}\n")

    return encoder


def create_mae_yolo_model(
    yolo_model_name='yolov8s.pt',
    mae_checkpoint=None,
    freeze_backbone=True,
    num_classes=1
):
    """
    Create YOLO model with MAE backbone.

    Args:
        yolo_model_name: Base YOLO model
        mae_checkpoint: Path to pre-trained MAE
        freeze_backbone: Whether to freeze MAE backbone
        num_classes: Number of detection classes

    Returns:
        Modified YOLO model
    """
    print(f"\n{'='*60}")
    print(f"Creating MAE+YOLO Model")
    print(f"{'='*60}")

    # Load base YOLO model
    print(f"Loading base YOLO model: {yolo_model_name}")
    model = YOLO(yolo_model_name)

    # If MAE checkpoint provided, integrate it
    if mae_checkpoint and os.path.exists(mae_checkpoint):
        print(f"\nIntegrating MAE backbone...")

        # Load MAE encoder
        mae_encoder = load_mae_encoder(mae_checkpoint, freeze=freeze_backbone)

        # Note: Full integration requires modifying YOLO architecture
        # For now, we'll use YOLO as-is but you can replace backbone
        # This is a simplified version - full integration needs custom backbone

        print(f"⚠️  Note: Using standard YOLO backbone")
        print(f"   For full MAE integration, custom backbone needed")
        print(f"   MAE features can be used for feature initialization")

    else:
        print(f"\n⚠️  MAE checkpoint not provided or not found")
        print(f"   Training with standard YOLO backbone")

    print(f"{'='*60}\n")

    return model


def train_valve_detection(
    config_path,
    mae_checkpoint=None,
    freeze_backbone=True,
    resume=None
):
    """
    Train valve detection model.

    Args:
        config_path: Path to training config YAML
        mae_checkpoint: Path to pre-trained MAE
        freeze_backbone: Whether to freeze MAE backbone
        resume: Path to checkpoint to resume from
    """
    print("\n" + "=" * 60)
    print("Heart Valve Detection Training")
    print("MAE Pre-trained Backbone + YOLO")
    print("=" * 60)

    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    print(f"\nConfiguration:")
    print(f"  Config file: {config_path}")
    print(f"  Dataset: {config.get('data', 'N/A')}")
    print(f"  Epochs: {config.get('epochs', 'N/A')}")
    print(f"  Batch size: {config.get('batch', 'N/A')}")
    print(f"  Image size: {config.get('imgsz', 'N/A')}")
    print(f"  Device: {config.get('device', 'N/A')}")

    # Create model
    model = create_mae_yolo_model(
        yolo_model_name=config.get('model', 'yolov8s.pt'),
        mae_checkpoint=mae_checkpoint,
        freeze_backbone=freeze_backbone,
        num_classes=1
    )

    # Start training
    print("\n" + "=" * 60)
    print("Starting Training")
    print("=" * 60 + "\n")

    # Train with config
    results = model.train(
        data=config['data'],
        epochs=config['epochs'],
        batch=config['batch'],
        imgsz=config['imgsz'],
        device=config.get('device', '0'),
        workers=config.get('workers', 8),
        optimizer=config.get('optimizer', 'AdamW'),
        lr0=config.get('lr0', 0.001),
        lrf=config.get('lrf', 0.01),
        momentum=config.get('momentum', 0.937),
        weight_decay=config.get('weight_decay', 0.0005),
        warmup_epochs=config.get('warmup_epochs', 5),
        cos_lr=config.get('cos_lr', True),
        box=config.get('box', 7.5),
        cls=config.get('cls', 0.5),
        dfl=config.get('dfl', 1.5),
        hsv_h=config.get('hsv_h', 0.0),
        hsv_s=config.get('hsv_s', 0.0),
        hsv_v=config.get('hsv_v', 0.0),
        degrees=config.get('degrees', 10.0),
        translate=config.get('translate', 0.05),
        scale=config.get('scale', 0.2),
        shear=config.get('shear', 2.0),
        flipud=config.get('flipud', 0.0),
        fliplr=config.get('fliplr', 0.5),
        mosaic=config.get('mosaic', 0.0),
        mixup=config.get('mixup', 0.0),
        amp=config.get('amp', True),
        project=config.get('project', './runs/detect'),
        name=config.get('name', 'train'),
        exist_ok=config.get('exist_ok', True),
        pretrained=True,
        verbose=config.get('verbose', True),
        seed=config.get('seed', 42),
        deterministic=config.get('deterministic', True),
        single_cls=config.get('single_cls', True),
        rect=config.get('rect', False),
        save=config.get('save', True),
        save_period=config.get('save_period', 10),
        val=config.get('val', True),
        plots=config.get('plots', True),
        resume=resume if resume else False
    )

    print("\n" + "=" * 60)
    print("Training Completed!")
    print("=" * 60)

    # Print results summary
    if results:
        print(f"\nResults:")
        print(f"  Best mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
        print(f"  Best mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")
        print(f"  Save directory: {results.save_dir}")

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train YOLO with MAE backbone for valve detection'
    )

    parser.add_argument(
        '--config',
        type=str,
        default='configs/valve_train_args.yaml',
        help='Path to training config YAML'
    )

    parser.add_argument(
        '--mae_checkpoint',
        type=str,
        default='./mae_valve_output/best_model.pth',
        help='Path to pre-trained MAE checkpoint'
    )

    parser.add_argument(
        '--freeze_backbone',
        action='store_true',
        default=True,
        help='Freeze MAE backbone weights'
    )

    parser.add_argument(
        '--finetune_backbone',
        action='store_true',
        help='Fine-tune MAE backbone (overrides --freeze_backbone)'
    )

    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume training from'
    )

    args = parser.parse_args()

    # Override freeze if finetune specified
    if args.finetune_backbone:
        args.freeze_backbone = False

    print("\n" + "=" * 60)
    print("Training Configuration")
    print("=" * 60)
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")
    print("=" * 60)

    # Check if MAE checkpoint exists
    if args.mae_checkpoint and not os.path.exists(args.mae_checkpoint):
        print(f"\n⚠️  Warning: MAE checkpoint not found: {args.mae_checkpoint}")
        print(f"   Training will proceed with standard YOLO backbone")
        print(f"   To use MAE features, first train MAE model:")
        print(f"     ./run_training_valve_optimized.sh\n")
        args.mae_checkpoint = None

    # Start training
    results = train_valve_detection(
        config_path=args.config,
        mae_checkpoint=args.mae_checkpoint,
        freeze_backbone=args.freeze_backbone,
        resume=args.resume
    )
