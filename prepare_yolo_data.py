"""
Prepare YOLO dataset from medical images.

This script prepares the dataset for YOLO training:
- Splits data into train (patient 1-30) and val (patient 31-40)
- Organizes into YOLO format: images/ and labels/
- Creates dataset YAML configuration
"""

import os
import shutil
from pathlib import Path
import yaml
from tqdm import tqdm


def create_yolo_dataset(
    source_dir,
    output_dir,
    train_patients=list(range(1, 31)) + list(range(41, 51)),  # patient0001-0030, 0041-0050
    val_patients=range(31, 41),   # patient0031 to patient0040
    labels_dir=None,
    copy_images=True
):
    """
    Prepare YOLO format dataset.

    Args:
        source_dir: Source directory with patient folders
        output_dir: Output directory for YOLO dataset
        train_patients: Range of patient IDs for training
        val_patients: Range of patient IDs for validation
        labels_dir: Directory containing label files (if available)
        copy_images: Whether to copy images (False = symlink)
    """
    source_path = Path(source_dir)
    output_path = Path(output_dir)

    # Create directory structure
    print("Creating YOLO directory structure...")
    for split in ['train', 'val']:
        (output_path / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_path / split / 'labels').mkdir(parents=True, exist_ok=True)

    # Process training data
    print("\nProcessing training data (patient 1-30)...")
    train_count = 0
    for patient_id in tqdm(train_patients, desc='Train'):
        patient_folder = source_path / f'patient{patient_id:04d}'

        if not patient_folder.exists():
            print(f"Warning: {patient_folder} not found, skipping...")
            continue

        # Copy/link images
        for img_file in patient_folder.glob('*.png'):
            dst_img = output_path / 'train' / 'images' / img_file.name

            if copy_images:
                shutil.copy2(img_file, dst_img)
            else:
                # Create symlink (faster, saves space)
                if dst_img.exists():
                    dst_img.unlink()
                dst_img.symlink_to(img_file.absolute())

            train_count += 1

            # Create empty label file (if labels_dir not provided)
            # You'll need to add actual annotations later
            label_file = output_path / 'train' / 'labels' / (img_file.stem + '.txt')
            if not label_file.exists() and labels_dir is None:
                label_file.touch()

    # Process validation data
    print("\nProcessing validation data (patient 31-40)...")
    val_count = 0
    for patient_id in tqdm(val_patients, desc='Val'):
        patient_folder = source_path / f'patient{patient_id:04d}'

        if not patient_folder.exists():
            print(f"Warning: {patient_folder} not found, skipping...")
            continue

        # Copy/link images
        for img_file in patient_folder.glob('*.png'):
            dst_img = output_path / 'val' / 'images' / img_file.name

            if copy_images:
                shutil.copy2(img_file, dst_img)
            else:
                if dst_img.exists():
                    dst_img.unlink()
                dst_img.symlink_to(img_file.absolute())

            val_count += 1

            # Create empty label file
            label_file = output_path / 'val' / 'labels' / (img_file.stem + '.txt')
            if not label_file.exists() and labels_dir is None:
                label_file.touch()

    print(f"\nDataset created:")
    print(f"  Training images: {train_count}")
    print(f"  Validation images: {val_count}")
    print(f"  Output directory: {output_path}")

    # Create dataset YAML
    create_dataset_yaml(output_path, train_count, val_count)

    return train_count, val_count


def create_dataset_yaml(output_dir, train_count, val_count):
    """
    Create YOLO dataset configuration YAML.

    Args:
        output_dir: Dataset root directory
        train_count: Number of training images
        val_count: Number of validation images
    """
    output_path = Path(output_dir)

    # YOLO dataset config
    config = {
        'path': str(output_path.absolute()),
        'train': str((output_path / 'train' / 'images').absolute()),
        'val': str((output_path / 'val' / 'images').absolute()),
        'nc': 1,  # Number of classes
        'names': ['aortic_valve']  # Class names
    }

    yaml_path = output_path / 'valve_detection.yaml'

    with open(yaml_path, 'w') as f:
        f.write('# ============================================\n')
        f.write('# Heart Valve Detection Dataset Config\n')
        f.write(f'# Generated for MAE+YOLO training\n')
        f.write('# ============================================\n\n')
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        f.write(f'\n# Statistics:\n')
        f.write(f'#   Train images: {train_count}\n')
        f.write(f'#   Val images: {val_count}\n')
        f.write(f'#   Train patients: 1-30, 41-50\n')
        f.write(f'#   Val patients: 31-40\n')

    print(f"\nDataset YAML created: {yaml_path}")

    return yaml_path


def copy_labels_if_available(labels_dir, yolo_dataset_dir):
    """
    Copy existing label files to YOLO dataset if available.

    Args:
        labels_dir: Directory containing label files
        yolo_dataset_dir: YOLO dataset directory
    """
    labels_path = Path(labels_dir)
    dataset_path = Path(yolo_dataset_dir)

    if not labels_path.exists():
        print(f"Labels directory not found: {labels_path}")
        return

    print(f"\nCopying labels from {labels_path}...")

    for split in ['train', 'val']:
        label_dst = dataset_path / split / 'labels'

        # Find and copy label files (recursively search in subdirectories)
        copied = 0
        for label_file in labels_path.rglob('*.txt'):
            # Check if corresponding image exists
            img_file = dataset_path / split / 'images' / (label_file.stem + '.png')

            if img_file.exists():
                shutil.copy2(label_file, label_dst / label_file.name)
                copied += 1

        print(f"  {split}: copied {copied} label files")


def validate_dataset(dataset_dir):
    """
    Validate YOLO dataset structure.

    Args:
        dataset_dir: Dataset root directory
    """
    dataset_path = Path(dataset_dir)

    print("\n" + "=" * 60)
    print("Dataset Validation")
    print("=" * 60)

    for split in ['train', 'val']:
        images_dir = dataset_path / split / 'images'
        labels_dir = dataset_path / split / 'labels'

        num_images = len(list(images_dir.glob('*.png')))
        num_labels = len(list(labels_dir.glob('*.txt')))

        print(f"\n{split.upper()}:")
        print(f"  Images: {num_images}")
        print(f"  Labels: {num_labels}")

        # Check if images and labels match
        if num_images != num_labels:
            print(f"  ⚠️  Warning: Image count != Label count")
        else:
            print(f"  ✅ Image-label pairs match")

        # Sample check: verify label files are not empty
        label_files = list(labels_dir.glob('*.txt'))
        if label_files:
            sample_label = label_files[0]
            with open(sample_label) as f:
                content = f.read().strip()

            if content:
                print(f"  ✅ Labels contain annotations")
            else:
                print(f"  ⚠️  Warning: Labels are empty (need annotation)")

    print("=" * 60)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Prepare YOLO dataset from medical images')

    parser.add_argument(
        '--source_dir',
        type=str,
        default='/DATA1/yunzhu/SSL/training_image/training_image',
        help='Source directory with patient folders'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default='/DATA1/yunzhu/SSL/yolo_dataset',
        help='Output directory for YOLO dataset'
    )

    parser.add_argument(
        '--labels_dir',
        type=str,
        default=None,
        help='Directory containing existing label files (optional)'
    )

    parser.add_argument(
        '--copy_images',
        action='store_true',
        default=False,
        help='Copy images instead of creating symlinks'
    )

    parser.add_argument(
        '--validate',
        action='store_true',
        help='Validate dataset after creation'
    )

    args = parser.parse_args()

    print("=" * 60)
    print("YOLO Dataset Preparation")
    print("=" * 60)
    print(f"Source: {args.source_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Train: patient 1-30, 41-50")
    print(f"Val: patient 31-40")
    print("=" * 60)

    # Create dataset
    train_count, val_count = create_yolo_dataset(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        copy_images=args.copy_images,
        labels_dir=args.labels_dir
    )

    # Copy labels if available
    if args.labels_dir:
        copy_labels_if_available(args.labels_dir, args.output_dir)

    # Validate
    if args.validate:
        validate_dataset(args.output_dir)

    print("\n✅ Dataset preparation complete!")
    print(f"\nNext steps:")
    print(f"1. Add annotations to label files in: {args.output_dir}/*/labels/")
    print(f"2. Update training config to use: {args.output_dir}/valve_detection.yaml")
    print(f"3. Start training with MAE backbone")
