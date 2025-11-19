"""
Train YOLO for Heart Valve Detection

Uses existing args.yaml configuration with MAE pre-trained features (optional).
Data split: patient 1-30 for training, patient 31-40 for validation.
"""

import os
import sys
import argparse
import yaml
import torch
from pathlib import Path
from ultralytics import YOLO


def update_config_paths(config, base_dir='/DATA1/yunzhu/SSL'):
    """
    Update paths in config to current environment.
    Only updates paths if they don't already exist (for backwards compatibility).

    Args:
        config: Config dictionary
        base_dir: Base directory

    Returns:
        Updated config
    """
    base_path = Path(base_dir)

    # Only update data path if it doesn't exist
    if 'data' in config:
        data_path = config['data']
        if not os.path.exists(data_path):
            # Try default path
            default_data = str(base_path / 'yolo_dataset' / 'valve_detection.yaml')
            if os.path.exists(default_data):
                config['data'] = default_data
    else:
        config['data'] = str(base_path / 'yolo_dataset' / 'valve_detection.yaml')

    # Only update project path if it's a Windows path or doesn't exist
    if 'project' in config:
        project_path = config['project']
        if 'C:\\' in project_path or '\\' in project_path:
            config['project'] = str(base_path / 'valve_training_results')
    else:
        config['project'] = str(base_path / 'valve_training_results')

    # Update save_dir if exists and is Windows path
    if 'save_dir' in config:
        save_dir = config['save_dir']
        if 'C:\\' in save_dir or '\\' in save_dir:
            config['save_dir'] = str(base_path / 'valve_training_results' / 'train')

    return config


def train_valve_detection(
    args_yaml='/DATA1/yunzhu/SSL/args.yaml',
    mae_checkpoint=None,
    resume=None
):
    """
    Train valve detection model.

    Args:
        args_yaml: Path to args.yaml configuration
        mae_checkpoint: Path to pre-trained MAE (optional)
        resume: Path to checkpoint to resume from
    """
    print("\n" + "=" * 80)
    print("Heart Valve Detection Training with YOLO")
    print("=" * 80)

    # Load config
    print(f"\nLoading configuration from: {args_yaml}")
    with open(args_yaml) as f:
        config = yaml.safe_load(f)

    # Update paths (only paths, keep all training parameters)
    config = update_config_paths(config)

    print(f"\n📝 Training Configuration:")
    print(f"  Model: {config.get('model')}")
    print(f"  Dataset: {config.get('data')}")
    print(f"  Epochs: {config.get('epochs')}")
    print(f"  Batch size: {config.get('batch')}")
    print(f"  Image size: {config.get('imgsz')}")
    print(f"  Device: {config.get('device')}")
    print(f"  Project: {config.get('project')}")

    # Check if dataset exists
    data_yaml = config['data']
    if not os.path.exists(data_yaml):
        print(f"\n⚠️  Dataset YAML not found: {data_yaml}")
        print(f"   Please run data preparation first:")
        print(f"     python prepare_yolo_data.py")
        return None

    # Load YOLO model
    model_name = config.get('model', 'yolov8s.pt')
    print(f"\n📦 Loading YOLO model: {model_name}")
    model = YOLO(model_name)

    # If MAE checkpoint provided, you can use it for feature initialization
    # (Full integration requires custom architecture)
    if mae_checkpoint and os.path.exists(mae_checkpoint):
        print(f"\n🔬 MAE checkpoint available: {mae_checkpoint}")
        print(f"   Note: Using standard YOLO architecture")
        print(f"   For full MAE integration, use custom backbone")
    else:
        print(f"\n📌 Training with standard YOLO backbone")

    # Create project directory
    os.makedirs(config['project'], exist_ok=True)

    # Save updated config
    updated_config_path = Path(config['project']) / 'training_config.yaml'
    with open(updated_config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"\n💾 Saved training config to: {updated_config_path}")

    # Start training
    print("\n" + "=" * 80)
    print("Starting Training...")
    print("=" * 80 + "\n")

    # Train using all parameters from config
    results = model.train(
        # Dataset
        data=config['data'],

        # Training duration
        epochs=config.get('epochs', 100),
        patience=config.get('patience', 20),

        # Batch and image size
        batch=config.get('batch', 32),
        imgsz=config.get('imgsz', 640),

        # Device and workers
        device=config.get('device', '0'),
        workers=config.get('workers', 8),

        # Optimizer settings
        optimizer=config.get('optimizer', 'AdamW'),
        lr0=config.get('lr0', 0.001),
        lrf=config.get('lrf', 0.01),
        momentum=config.get('momentum', 0.937),
        weight_decay=config.get('weight_decay', 0.0005),
        warmup_epochs=config.get('warmup_epochs', 3),
        warmup_momentum=config.get('warmup_momentum', 0.8),
        warmup_bias_lr=config.get('warmup_bias_lr', 0.1),

        # Loss weights
        box=config.get('box', 12.0),
        cls=config.get('cls', 0.2),
        dfl=config.get('dfl', 2.2),

        # Learning rate scheduler
        cos_lr=config.get('cos_lr', True),
        close_mosaic=config.get('close_mosaic', 30),

        # Augmentation
        hsv_h=config.get('hsv_h', 0.0),
        hsv_s=config.get('hsv_s', 0.0),
        hsv_v=config.get('hsv_v', 0.0),
        degrees=config.get('degrees', 5.0),
        translate=config.get('translate', 0.04),
        scale=config.get('scale', 0.2),
        shear=config.get('shear', 1.0),
        perspective=config.get('perspective', 0.0),
        flipud=config.get('flipud', 0.0),
        fliplr=config.get('fliplr', 0.0),
        mosaic=config.get('mosaic', 0.0),
        mixup=config.get('mixup', 0.0),
        copy_paste=config.get('copy_paste', 0.0),
        auto_augment=config.get('auto_augment', 'randaugment'),
        erasing=config.get('erasing', 0.0),

        # Training settings
        amp=config.get('amp', True),
        fraction=config.get('fraction', 1.0),
        pretrained=config.get('pretrained', True),
        seed=config.get('seed', 0),
        deterministic=config.get('deterministic', True),
        single_cls=config.get('single_cls', True),
        rect=config.get('rect', False),

        # Saving
        save=config.get('save', True),
        save_period=config.get('save_period', 20),
        cache=config.get('cache', False),

        # Validation
        val=config.get('val', True),
        plots=config.get('plots', True),

        # Output
        project=config['project'],
        name=config.get('name', 'train'),
        exist_ok=config.get('exist_ok', True),
        verbose=config.get('verbose', True),

        # Resume
        resume=resume if resume else config.get('resume', False)
    )

    print("\n" + "=" * 80)
    print("✅ Training Completed!")
    print("=" * 80)

    # Print results
    if results:
        print(f"\n📊 Training Results:")
        results_dict = results.results_dict
        if results_dict:
            print(f"  mAP50: {results_dict.get('metrics/mAP50(B)', 'N/A')}")
            print(f"  mAP50-95: {results_dict.get('metrics/mAP50-95(B)', 'N/A')}")
        print(f"  Output directory: {results.save_dir}")

        # List saved models
        save_dir = Path(results.save_dir) / 'weights'
        if save_dir.exists():
            models = list(save_dir.glob('*.pt'))
            if models:
                print(f"\n📦 Saved Models:")
                for model_path in models:
                    size_mb = model_path.stat().st_size / (1024 * 1024)
                    print(f"    {model_path.name} ({size_mb:.1f} MB)")

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train YOLO for heart valve detection'
    )

    parser.add_argument(
        '--args_yaml',
        type=str,
        default='/DATA1/yunzhu/SSL/args.yaml',
        help='Path to args.yaml configuration'
    )

    parser.add_argument(
        '--mae_checkpoint',
        type=str,
        default='./mae_valve_output/best_model.pth',
        help='Path to pre-trained MAE checkpoint (optional)'
    )

    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume training from'
    )

    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("Configuration")
    print("=" * 80)
    print(f"  Args YAML: {args.args_yaml}")
    print(f"  MAE checkpoint: {args.mae_checkpoint}")
    print(f"  Resume: {args.resume}")
    print("=" * 80)

    # Check if args.yaml exists
    if not os.path.exists(args.args_yaml):
        print(f"\n❌ Error: args.yaml not found at {args.args_yaml}")
        sys.exit(1)

    # Start training
    results = train_valve_detection(
        args_yaml=args.args_yaml,
        mae_checkpoint=args.mae_checkpoint if os.path.exists(args.mae_checkpoint or '') else None,
        resume=args.resume
    )
