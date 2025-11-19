import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
import torchvision.transforms as transforms
import numpy as np


class MAEMedicalImageDataset(Dataset):
    """
    Dataset for MAE training with medical images.
    Uses minimal augmentation to preserve fine details important for valve detection.
    """

    def __init__(self, root_dir, transform=None, image_size=224):
        """
        Args:
            root_dir (str): Path to the directory containing patient folders
            transform: Custom augmentation transforms (optional)
            image_size: Size to resize images to
        """
        self.root_dir = Path(root_dir)
        self.image_size = image_size
        self.image_paths = []

        # Collect all image paths
        for patient_folder in sorted(self.root_dir.glob('patient*')):
            if patient_folder.is_dir():
                for img_path in sorted(patient_folder.glob('*.png')):
                    self.image_paths.append(img_path)

        print(f"Found {len(self.image_paths)} images in {root_dir}")

        # Set up transform
        if transform is None:
            # Minimal augmentation to preserve fine details
            self.transform = self.get_default_transform(image_size)
        else:
            self.transform = transform

    def get_default_transform(self, image_size):
        """
        Create default transform with minimal augmentation.
        Designed to preserve fine details for valve detection.
        """
        return transforms.Compose([
            # Resize to target size
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),

            # Minimal geometric augmentation
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),

            # Slight rotation (smaller range to preserve details)
            transforms.RandomRotation(degrees=10),

            # Convert grayscale to RGB (for ViT compatibility)
            transforms.Lambda(lambda x: x.convert('RGB')),

            # To tensor
            transforms.ToTensor(),

            # Normalize (ImageNet stats work well for medical images too)
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        # Load image
        image = Image.open(img_path).convert('L')  # Load as grayscale

        # Apply transform
        if self.transform:
            image = self.transform(image)

        return image


class MAEValveDetectionTransform:
    """
    Specialized augmentation for valve detection tasks.
    Uses very minimal augmentation to preserve critical details.
    """

    def __init__(self, image_size=224, preserve_details=True):
        """
        Args:
            image_size: Size to resize images to
            preserve_details: If True, use minimal augmentation
        """
        if preserve_details:
            # Minimal augmentation for preserving valve details
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size),
                                interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(p=0.3),
                transforms.Lambda(lambda x: x.convert('RGB')),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225]),
            ])
        else:
            # Standard augmentation
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size),
                                interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.RandomAffine(
                    degrees=0,
                    translate=(0.05, 0.05),
                    scale=(0.95, 1.05)
                ),
                transforms.Lambda(lambda x: x.convert('RGB')),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225]),
            ])

    def __call__(self, x):
        return self.transform(x)


def get_mae_dataloaders(
    train_dir,
    batch_size=64,
    num_workers=4,
    image_size=224,
    preserve_details=True
):
    """
    Create dataloaders for MAE training.

    Args:
        train_dir: Path to training images directory
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
        image_size: Size to resize images to
        preserve_details: Use minimal augmentation to preserve fine details

    Returns:
        train_loader: DataLoader for training
    """
    # Create transform
    transform = MAEValveDetectionTransform(
        image_size=image_size,
        preserve_details=preserve_details
    )

    # Create dataset
    train_dataset = MAEMedicalImageDataset(
        root_dir=train_dir,
        transform=transform,
        image_size=image_size
    )

    # Create dataloader
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    return train_loader


class MAEVisualizationDataset(Dataset):
    """
    Dataset for visualizing MAE reconstructions.
    Returns original images without augmentation.
    """

    def __init__(self, root_dir, image_size=224, max_samples=None):
        self.root_dir = Path(root_dir)
        self.image_size = image_size
        self.image_paths = []

        # Collect image paths
        for patient_folder in sorted(self.root_dir.glob('patient*')):
            if patient_folder.is_dir():
                for img_path in sorted(patient_folder.glob('*.png')):
                    self.image_paths.append(img_path)
                    if max_samples and len(self.image_paths) >= max_samples:
                        break
            if max_samples and len(self.image_paths) >= max_samples:
                break

        print(f"Loaded {len(self.image_paths)} images for visualization")

        # Simple transform without augmentation
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size),
                            interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.Lambda(lambda x: x.convert('RGB')),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('L')

        if self.transform:
            image = self.transform(image)

        return image, str(img_path)


def denormalize_image(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Denormalize image tensor for visualization.

    Args:
        img: Normalized image tensor (C, H, W) or (B, C, H, W)
        mean: Mean used for normalization
        std: Std used for normalization

    Returns:
        Denormalized image
    """
    if len(img.shape) == 4:  # Batch
        mean = torch.tensor(mean).view(1, 3, 1, 1).to(img.device)
        std = torch.tensor(std).view(1, 3, 1, 1).to(img.device)
    else:  # Single image
        mean = torch.tensor(mean).view(3, 1, 1).to(img.device)
        std = torch.tensor(std).view(3, 1, 1).to(img.device)

    img = img * std + mean
    img = torch.clamp(img, 0, 1)
    return img
