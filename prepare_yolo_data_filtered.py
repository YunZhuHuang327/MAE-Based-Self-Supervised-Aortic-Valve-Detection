"""
Prepare YOLO dataset with only labeled (positive) images.

This script creates a dataset containing ONLY images that have annotations,
ensuring all training/validation samples contain heart valves.
"""

import os
import shutil
from pathlib import Path
import yaml
from tqdm import tqdm


def create_filtered_yolo_dataset(
    source_images_dir,
    source_labels_dir,
    output_dir,
    train_patients=list(range(1, 31)) + list(range(41, 51)),  # patient0001-0030, 0041-0050
    val_patients=range(31, 41),   # patient0031 to patient0040
    copy_images=True
):
    """
    Prepare YOLO format dataset with only labeled images.

    Args:
        source_images_dir: Source directory with patient image folders
        source_labels_dir: Source directory with patient label folders
        output_dir: Output directory for YOLO dataset
        train_patients: Patient IDs for training
        val_patients: Patient IDs for validation
        copy_images: Whether to copy images (False = symlink)
    """
    source_img_path = Path(source_images_dir)
    source_lbl_path = Path(source_labels_dir)
    output_path = Path(output_dir)

    # Create directory structure
    print("Creating YOLO directory structure...")
    for split in ['train', 'val']:
        (output_path / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_path / split / 'labels').mkdir(parents=True, exist_ok=True)

    # Process training data
    print("\nProcessing training data (patient 1-30, 41-50)...")
    train_count = 0
    for patient_id in tqdm(train_patients, desc='Train'):
        patient_img_folder = source_img_path / f'patient{patient_id:04d}'
        patient_lbl_folder = source_lbl_path / f'patient{patient_id:04d}'

        if not patient_img_folder.exists():
            print(f"Warning: {patient_img_folder} not found, skipping...")
            continue

        if not patient_lbl_folder.exists():
            print(f"Warning: {patient_lbl_folder} not found, skipping...")
            continue

        # Find all label files for this patient
        label_files = list(patient_lbl_folder.glob('*.txt'))

        for label_file in label_files:
            # Check if label is non-empty
            if label_file.stat().st_size == 0:
                continue

            # Find corresponding image
            img_file = patient_img_folder / (label_file.stem + '.png')

            if not img_file.exists():
                continue

            # Copy/link image
            dst_img = output_path / 'train' / 'images' / img_file.name

            if copy_images:
                shutil.copy2(img_file, dst_img)
            else:
                if dst_img.exists():
                    dst_img.unlink()
                dst_img.symlink_to(img_file.absolute())

            # Copy label
            dst_label = output_path / 'train' / 'labels' / label_file.name
            shutil.copy2(label_file, dst_label)

            train_count += 1

    # Process validation data
    print("\nProcessing validation data (patient 31-40)...")
    val_count = 0
    for patient_id in tqdm(val_patients, desc='Val'):
        patient_img_folder = source_img_path / f'patient{patient_id:04d}'
        patient_lbl_folder = source_lbl_path / f'patient{patient_id:04d}'

        if not patient_img_folder.exists():
            print(f"Warning: {patient_img_folder} not found, skipping...")
            continue

        if not patient_lbl_folder.exists():
            print(f"Warning: {patient_lbl_folder} not found, skipping...")
            continue

        # Find all label files for this patient
        label_files = list(patient_lbl_folder.glob('*.txt'))

        for label_file in label_files:
            # Check if label is non-empty
            if label_file.stat().st_size == 0:
                continue

            # Find corresponding image
            img_file = patient_img_folder / (label_file.stem + '.png')

            if not img_file.exists():
                continue

            # Copy/link image
            dst_img = output_path / 'val' / 'images' / img_file.name

            if copy_images:
                shutil.copy2(img_file, dst_img)
            else:
                if dst_img.exists():
                    dst_img.unlink()
                dst_img.symlink_to(img_file.absolute())

            # Copy label
            dst_label = output_path / 'val' / 'labels' / label_file.name
            shutil.copy2(label_file, dst_label)

            val_count += 1

    print(f"\nFiltered dataset created:")
    print(f"  Training samples (with labels): {train_count}")
    print(f"  Validation samples (with labels): {val_count}")
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
        f.write('# FILTERED: Only labeled (positive) samples\n')
        f.write('# ============================================\n\n')
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        f.write(f'\n# Statistics:\n')
        f.write(f'#   Train samples: {train_count} (all positive)\n')
        f.write(f'#   Val samples: {val_count} (all positive)\n')
        f.write(f'#   Train patients: 1-30, 41-50\n')
        f.write(f'#   Val patients: 31-40\n')

    print(f"\nDataset YAML created: {yaml_path}")

    return yaml_path


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
            print(f"  ✅ Image-label pairs match perfectly")

        # Check for empty labels
        empty_count = 0
        for label_file in labels_dir.glob('*.txt'):
            if label_file.stat().st_size == 0:
                empty_count += 1

        if empty_count > 0:
            print(f"  ⚠️  Warning: {empty_count} empty label files")
        else:
            print(f"  ✅ All labels contain annotations")

    print("=" * 60)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Prepare filtered YOLO dataset (only labeled images)'
    )

    parser.add_argument(
        '--source_images',
        type=str,
        default='/DATA1/yunzhu/SSL/training_image/training_image',
        help='Source directory with patient image folders'
    )

    parser.add_argument(
        '--source_labels',
        type=str,
        default='/DATA1/yunzhu/SSL/training_label/training_label',
        help='Source directory with patient label folders'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default='/DATA1/yunzhu/SSL/yolo_dataset',
        help='Output directory for YOLO dataset'
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
    print("YOLO Filtered Dataset Preparation")
    print("(Only labeled/positive samples)")
    print("=" * 60)
    print(f"Source images: {args.source_images}")
    print(f"Source labels: {args.source_labels}")
    print(f"Output: {args.output_dir}")
    print(f"Train: patient 1-30, 41-50")
    print(f"Val: patient 31-40 (all positive)")
    print("=" * 60)

    # Create filtered dataset
    train_count, val_count = create_filtered_yolo_dataset(
        source_images_dir=args.source_images,
        source_labels_dir=args.source_labels,
        output_dir=args.output_dir,
        copy_images=args.copy_images
    )

    # Validate
    if args.validate:
        validate_dataset(args.output_dir)

    print("\n✅ Filtered dataset preparation complete!")
    print(f"\nDataset statistics:")
    print(f"  Training: {train_count} labeled samples")
    print(f"  Validation: {val_count} labeled samples")
    print(f"  All samples contain heart valve annotations")
