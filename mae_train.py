import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import argparse
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from mae_dataset import get_mae_dataloaders, MAEVisualizationDataset, denormalize_image
from mae_model import build_mae_model


def visualize_reconstruction(model, dataset, device, save_path, num_samples=8):
    """
    Visualize MAE reconstructions.

    Args:
        model: Trained MAE model
        dataset: Visualization dataset
        device: Device to run on
        save_path: Path to save visualization
        num_samples: Number of samples to visualize
    """
    model.eval()

    # Get random samples
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)

    fig, axes = plt.subplots(3, num_samples, figsize=(num_samples * 3, 9))

    with torch.no_grad():
        for i, idx in enumerate(indices):
            img, _ = dataset[idx]
            img = img.unsqueeze(0).to(device)

            # Forward pass
            loss, pred, mask = model(img)

            # Unpatchify prediction
            pred_img = model.unpatchify(pred)

            # Denormalize
            img_vis = denormalize_image(img[0].cpu())
            pred_vis = denormalize_image(pred_img[0].cpu())

            # Create masked image
            mask_vis = mask[0].cpu().numpy()
            mask_vis = mask_vis.reshape(int(np.sqrt(len(mask_vis))), -1)
            mask_vis = np.repeat(np.repeat(mask_vis, model.patch_size, axis=0),
                                model.patch_size, axis=1)
            mask_vis = np.stack([mask_vis] * 3, axis=0)

            img_masked = img_vis.numpy() * mask_vis
            img_masked = torch.from_numpy(img_masked)

            # Convert to displayable format (use only first channel for grayscale)
            img_display = img_vis[0].numpy()
            img_masked_display = img_masked[0].numpy()
            pred_display = pred_vis[0].numpy()

            # Plot
            axes[0, i].imshow(img_display, cmap='gray')
            axes[0, i].axis('off')
            if i == 0:
                axes[0, i].set_title('Original', fontsize=10)

            axes[1, i].imshow(img_masked_display, cmap='gray')
            axes[1, i].axis('off')
            if i == 0:
                axes[1, i].set_title(f'Masked ({int(model.mask_ratio*100)}%)', fontsize=10)

            axes[2, i].imshow(pred_display, cmap='gray')
            axes[2, i].axis('off')
            if i == 0:
                axes[2, i].set_title('Reconstruction', fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to {save_path}")


def train_one_epoch(model, dataloader, optimizer, device, epoch, scaler=None):
    """
    Train for one epoch.

    Args:
        model: MAE model
        dataloader: Training dataloader
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number
        scaler: GradScaler for mixed precision training

    Returns:
        avg_loss: Average loss for the epoch
    """
    model.train()
    total_loss = 0
    num_batches = 0

    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch}')

    for batch_idx, images in enumerate(progress_bar):
        # Move to device
        images = images.to(device)

        # Forward pass with mixed precision
        if scaler is not None:
            with torch.amp.autocast('cuda'):
                loss, _, _ = model(images)

            # Backward pass
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard training
            loss, _, _ = model(images)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Update statistics
        total_loss += loss.item()
        num_batches += 1

        # Update progress bar
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / num_batches
    return avg_loss


def save_checkpoint(model, optimizer, epoch, loss, filepath, scaler=None):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    if scaler is not None:
        checkpoint['scaler_state_dict'] = scaler.state_dict()

    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def train_mae(args):
    """
    Main training function for MAE.

    Args:
        args: Command line arguments
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    vis_dir = os.path.join(args.output_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)

    # Create tensorboard writer
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'logs'))

    # Get dataloader
    print("Loading data...")
    train_loader = get_mae_dataloaders(
        train_dir=args.train_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        preserve_details=args.preserve_details
    )

    # Create visualization dataset
    vis_dataset = MAEVisualizationDataset(
        root_dir=args.train_dir,
        image_size=args.image_size,
        max_samples=16
    )

    # Create model
    print(f"Creating MAE model...")
    print(f"  - Image size: {args.image_size}")
    print(f"  - Patch size: {args.patch_size}")
    print(f"  - Mask ratio: {args.mask_ratio}")
    print(f"  - Encoder depth: {args.encoder_depth}")
    print(f"  - Decoder depth: {args.decoder_depth}")

    model = build_mae_model(
        img_size=args.image_size,
        patch_size=args.patch_size,
        in_chans=3,
        embed_dim=args.embed_dim,
        depth=args.encoder_depth,
        num_heads=args.num_heads,
        decoder_embed_dim=args.decoder_embed_dim,
        decoder_depth=args.decoder_depth,
        decoder_num_heads=args.decoder_num_heads,
        mask_ratio=args.mask_ratio
    )
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay
    )

    # Learning rate scheduler with warmup
    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        else:
            progress = (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)
            return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Mixed precision training
    scaler = torch.amp.GradScaler('cuda') if args.use_amp and device.type == 'cuda' else None

    # Load checkpoint if specified
    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint from {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            if scaler is not None and 'scaler_state_dict' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
            print(f"Resumed from epoch {start_epoch}")
        else:
            print(f"No checkpoint found at {args.resume}")

    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    print("=" * 80)
    best_loss = float('inf')

    for epoch in range(start_epoch, args.epochs):
        # Train one epoch
        avg_loss = train_one_epoch(
            model, train_loader, optimizer, device, epoch, scaler
        )

        # Log to tensorboard
        writer.add_scalar('Loss/train', avg_loss, epoch)
        writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], epoch)

        print(f"Epoch {epoch}/{args.epochs-1} - Loss: {avg_loss:.4f} - LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Update learning rate
        scheduler.step()

        # Visualize reconstructions
        if (epoch + 1) % args.vis_freq == 0:
            vis_path = os.path.join(vis_dir, f'reconstruction_epoch_{epoch}.png')
            visualize_reconstruction(model, vis_dataset, device, vis_path)

        # Save checkpoint
        if (epoch + 1) % args.save_freq == 0:
            checkpoint_path = os.path.join(
                args.output_dir,
                f'checkpoint_epoch_{epoch}.pth'
            )
            save_checkpoint(model, optimizer, epoch, avg_loss, checkpoint_path, scaler)

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_path = os.path.join(args.output_dir, 'best_model.pth')
            save_checkpoint(model, optimizer, epoch, avg_loss, best_model_path, scaler)

    # Save final model
    final_model_path = os.path.join(args.output_dir, 'final_model.pth')
    save_checkpoint(model, optimizer, args.epochs-1, avg_loss, final_model_path, scaler)

    # Final visualization
    final_vis_path = os.path.join(vis_dir, 'final_reconstruction.png')
    visualize_reconstruction(model, vis_dataset, device, final_vis_path, num_samples=8)

    writer.close()
    print("\n" + "=" * 80)
    print(f"Training completed! Best loss: {best_loss:.4f}")
    print(f"Models saved in {args.output_dir}")
    print(f"Visualizations saved in {vis_dir}")
    print("=" * 80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Self-Supervised Learning with MAE for Valve Detection')

    # Data parameters
    parser.add_argument('--train_dir', type=str,
                       default='/DATA1/yunzhu/SSL/training_image/training_image',
                       help='Path to training images directory')
    parser.add_argument('--image_size', type=int, default=224,
                       help='Size to resize images to')
    parser.add_argument('--preserve_details', action='store_true', default=True,
                       help='Use minimal augmentation to preserve fine details')

    # Model parameters
    parser.add_argument('--patch_size', type=int, default=16,
                       help='Patch size for MAE')
    parser.add_argument('--mask_ratio', type=float, default=0.75,
                       help='Ratio of patches to mask')
    parser.add_argument('--embed_dim', type=int, default=768,
                       help='Encoder embedding dimension')
    parser.add_argument('--encoder_depth', type=int, default=12,
                       help='Encoder depth (number of transformer blocks)')
    parser.add_argument('--num_heads', type=int, default=12,
                       help='Number of attention heads in encoder')
    parser.add_argument('--decoder_embed_dim', type=int, default=512,
                       help='Decoder embedding dimension')
    parser.add_argument('--decoder_depth', type=int, default=8,
                       help='Decoder depth (number of transformer blocks)')
    parser.add_argument('--decoder_num_heads', type=int, default=16,
                       help='Number of attention heads in decoder')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=200,
                       help='Number of training epochs')
    parser.add_argument('--warmup_epochs', type=int, default=10,
                       help='Number of warmup epochs')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1.5e-4,
                       help='Base learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                       help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--use_amp', action='store_true', default=True,
                       help='Use automatic mixed precision training')

    # Save parameters
    parser.add_argument('--output_dir', type=str, default='./mae_output',
                       help='Directory to save models and logs')
    parser.add_argument('--save_freq', type=int, default=20,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--vis_freq', type=int, default=10,
                       help='Visualize reconstructions every N epochs')
    parser.add_argument('--resume', type=str, default='',
                       help='Path to checkpoint to resume from')

    args = parser.parse_args()

    # Print configuration
    print("=" * 80)
    print("MAE Self-Supervised Learning for Heart Valve Detection")
    print("=" * 80)
    for arg in vars(args):
        print(f"{arg:25s}: {getattr(args, arg)}")
    print("=" * 80)

    # Start training
    train_mae(args)
