import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from tqdm import tqdm
from mae_dataset import MAEVisualizationDataset, denormalize_image
from mae_model import build_mae_model


class MAEFeatureExtractor(nn.Module):
    """
    Wrapper to extract features from MAE encoder.
    Useful for downstream tasks like valve detection.
    """

    def __init__(self, mae_model):
        super().__init__()
        self.encoder = mae_model.encoder
        self.patch_embed = mae_model.encoder.patch_embed

    def forward(self, x):
        """
        Extract features without masking.

        Args:
            x: Input images (B, C, H, W)

        Returns:
            cls_token: CLS token features (B, embed_dim)
            patch_tokens: All patch tokens (B, num_patches, embed_dim)
        """
        # Patch embedding
        x = self.patch_embed(x)

        # Add positional embeddings
        x = x + self.encoder.pos_embed[:, 1:, :]

        # Add cls token
        cls_token = self.encoder.cls_token + self.encoder.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Apply transformer blocks
        for block in self.encoder.blocks:
            x = block(x)

        x = self.encoder.norm(x)

        # Separate cls token and patch tokens
        cls_token = x[:, 0]
        patch_tokens = x[:, 1:]

        return cls_token, patch_tokens


def extract_features(model, dataloader, device, use_cls_token=True):
    """
    Extract features from the encoder for all images.

    Args:
        model: Trained MAE model
        dataloader: DataLoader for images
        device: Device to run on
        use_cls_token: If True, use CLS token; otherwise use mean of patch tokens

    Returns:
        features: Extracted features [N, feature_dim]
        image_paths: Corresponding image paths
    """
    feature_extractor = MAEFeatureExtractor(model).to(device)
    feature_extractor.eval()

    all_features = []
    all_paths = []

    with torch.no_grad():
        for images, paths in tqdm(dataloader, desc='Extracting features'):
            images = images.to(device)

            # Get features
            cls_token, patch_tokens = feature_extractor(images)

            if use_cls_token:
                features = cls_token
            else:
                features = patch_tokens.mean(dim=1)

            all_features.append(features.cpu())
            all_paths.extend(paths)

    all_features = torch.cat(all_features, dim=0)
    return all_features.numpy(), all_paths


def visualize_embeddings(features, labels, save_path='embeddings_tsne.png', method='tsne'):
    """
    Visualize embeddings using t-SNE or PCA.

    Args:
        features: Feature vectors [N, feature_dim]
        labels: Labels for coloring (e.g., patient IDs)
        save_path: Path to save visualization
        method: 'tsne' or 'pca'
    """
    print(f"Computing {method.upper()}...")

    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
    elif method == 'pca':
        reducer = PCA(n_components=2, random_state=42)
    else:
        raise ValueError(f"Unknown method: {method}")

    embeddings_2d = reducer.fit_transform(features)

    plt.figure(figsize=(14, 12))
    scatter = plt.scatter(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        c=labels,
        cmap='tab20',
        alpha=0.6,
        s=20
    )
    plt.colorbar(scatter, label='Patient ID')
    plt.title(f'{method.upper()} Visualization of MAE Learned Features')
    plt.xlabel(f'{method.upper()} Dimension 1')
    plt.ylabel(f'{method.upper()} Dimension 2')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to {save_path}")


def compute_feature_statistics(features, labels):
    """
    Compute statistics about the learned features.

    Args:
        features: Feature vectors [N, feature_dim]
        labels: Patient labels
    """
    print("\n" + "=" * 80)
    print("Feature Statistics")
    print("=" * 80)

    # Feature dimensionality
    print(f"Feature dimension: {features.shape[1]}")
    print(f"Number of samples: {features.shape[0]}")

    # Feature norms
    feature_norms = np.linalg.norm(features, axis=1)
    print(f"\nFeature norm - Mean: {feature_norms.mean():.4f}, Std: {feature_norms.std():.4f}")
    print(f"Feature norm - Min: {feature_norms.min():.4f}, Max: {feature_norms.max():.4f}")

    # Feature value statistics
    print(f"\nFeature values - Mean: {features.mean():.4f}, Std: {features.std():.4f}")
    print(f"Feature values - Min: {features.min():.4f}, Max: {features.max():.4f}")

    # Intra-patient similarity
    unique_labels = np.unique(labels)
    intra_similarities = []

    print(f"\nAnalyzing {len(unique_labels)} patients...")

    for label in unique_labels:
        mask = labels == label
        patient_features = features[mask]

        if len(patient_features) > 1:
            # Compute pairwise similarities within patient
            from sklearn.metrics.pairwise import cosine_similarity
            sim_matrix = cosine_similarity(patient_features)
            # Get upper triangle (excluding diagonal)
            triu_indices = np.triu_indices_from(sim_matrix, k=1)
            if len(triu_indices[0]) > 0:
                intra_similarities.extend(sim_matrix[triu_indices])

    if intra_similarities:
        print(f"\nIntra-patient similarity (cosine):")
        print(f"  Mean: {np.mean(intra_similarities):.4f}")
        print(f"  Std: {np.std(intra_similarities):.4f}")
        print(f"  Min: {np.min(intra_similarities):.4f}")
        print(f"  Max: {np.max(intra_similarities):.4f}")

    # Inter-patient similarity
    if len(unique_labels) > 1:
        from sklearn.metrics.pairwise import cosine_similarity

        # Compute mean feature per patient
        patient_means = []
        for label in unique_labels:
            mask = labels == label
            patient_means.append(features[mask].mean(axis=0))

        patient_means = np.array(patient_means)
        inter_sim_matrix = cosine_similarity(patient_means)

        # Get upper triangle (excluding diagonal)
        triu_indices = np.triu_indices_from(inter_sim_matrix, k=1)
        inter_similarities = inter_sim_matrix[triu_indices]

        print(f"\nInter-patient similarity (cosine):")
        print(f"  Mean: {inter_similarities.mean():.4f}")
        print(f"  Std: {inter_similarities.std():.4f}")
        print(f"  Min: {inter_similarities.min():.4f}")
        print(f"  Max: {inter_similarities.max():.4f}")

        # Separation metric
        if intra_similarities:
            separation = np.mean(intra_similarities) - inter_similarities.mean()
            print(f"\nSeparation (Intra - Inter): {separation:.4f}")
            print(f"  Higher values indicate better patient clustering")

    print("=" * 80)


def visualize_attention_maps(model, dataset, device, save_dir, num_samples=4):
    """
    Visualize attention maps from the encoder.

    Args:
        model: Trained MAE model
        dataset: Visualization dataset
        device: Device to run on
        save_dir: Directory to save visualizations
        num_samples: Number of samples to visualize
    """
    os.makedirs(save_dir, exist_ok=True)

    model.eval()
    feature_extractor = MAEFeatureExtractor(model).to(device)

    # Get random samples
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)

    print(f"\nGenerating attention visualizations...")

    for idx in tqdm(indices):
        img, path = dataset[idx]
        img = img.unsqueeze(0).to(device)

        with torch.no_grad():
            # Get features
            cls_token, patch_tokens = feature_extractor(img)

            # Get attention from last layer
            # Note: This is simplified - actual attention requires modifying the model
            # to return attention weights

        # For now, visualize patch token activations
        patch_tokens_np = patch_tokens[0].cpu().numpy()
        patch_mean = patch_tokens_np.mean(axis=-1)

        # Reshape to grid
        grid_size = int(np.sqrt(len(patch_mean)))
        attention_map = patch_mean.reshape(grid_size, grid_size)

        # Original image
        img_vis = denormalize_image(img[0].cpu())
        img_display = img_vis[0].numpy()

        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        axes[0].imshow(img_display, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        im = axes[1].imshow(attention_map, cmap='hot', interpolation='bilinear')
        axes[1].set_title('Patch Token Activation')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1])

        plt.tight_layout()
        save_path = os.path.join(save_dir, f'attention_{idx}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    print(f"Attention visualizations saved to {save_dir}")


def evaluate_model(args):
    """
    Main evaluation function.

    Args:
        args: Command line arguments
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    print(f"\nLoading model from {args.checkpoint}")
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

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"Model loaded from epoch {checkpoint['epoch']}")
    print(f"Training loss: {checkpoint['loss']:.4f}")

    # Create dataset and dataloader
    print(f"\nLoading data from {args.data_dir}")
    eval_dataset = MAEVisualizationDataset(
        root_dir=args.data_dir,
        image_size=args.image_size,
        max_samples=args.max_samples
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Extract features
    print("\nExtracting features...")
    features, image_paths = extract_features(
        model, eval_loader, device, use_cls_token=args.use_cls_token
    )

    # Extract patient labels from paths
    labels = []
    for path in image_paths:
        # Extract patient ID from path
        patient_id = os.path.basename(os.path.dirname(path))
        label = int(patient_id.replace('patient', ''))
        labels.append(label)
    labels = np.array(labels)

    # Compute statistics
    compute_feature_statistics(features, labels)

    # Save features
    if args.save_features:
        features_path = os.path.join(args.output_dir, 'mae_features.npy')
        labels_path = os.path.join(args.output_dir, 'labels.npy')
        paths_path = os.path.join(args.output_dir, 'image_paths.txt')

        np.save(features_path, features)
        np.save(labels_path, labels)
        with open(paths_path, 'w') as f:
            f.write('\n'.join(image_paths))

        print(f"\nFeatures saved to {features_path}")
        print(f"Labels saved to {labels_path}")
        print(f"Image paths saved to {paths_path}")

    # Visualize embeddings
    if args.visualize:
        print("\nCreating visualizations...")

        # t-SNE
        tsne_path = os.path.join(args.output_dir, 'embeddings_tsne.png')
        visualize_embeddings(features, labels, tsne_path, method='tsne')

        # PCA
        pca_path = os.path.join(args.output_dir, 'embeddings_pca.png')
        visualize_embeddings(features, labels, pca_path, method='pca')

    # Visualize attention
    if args.visualize_attention:
        attention_dir = os.path.join(args.output_dir, 'attention_maps')
        visualize_attention_maps(model, eval_dataset, device, attention_dir)

    print("\n" + "=" * 80)
    print("Evaluation completed!")
    print(f"Results saved in {args.output_dir}")
    print("=" * 80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate MAE Model')

    # Model parameters
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--patch_size', type=int, default=16,
                       help='Patch size')
    parser.add_argument('--mask_ratio', type=float, default=0.75,
                       help='Mask ratio (for info only)')
    parser.add_argument('--embed_dim', type=int, default=768,
                       help='Encoder embedding dimension')
    parser.add_argument('--encoder_depth', type=int, default=12,
                       help='Encoder depth')
    parser.add_argument('--num_heads', type=int, default=12,
                       help='Number of attention heads in encoder')
    parser.add_argument('--decoder_embed_dim', type=int, default=512,
                       help='Decoder embedding dimension')
    parser.add_argument('--decoder_depth', type=int, default=8,
                       help='Decoder depth')
    parser.add_argument('--decoder_num_heads', type=int, default=16,
                       help='Number of attention heads in decoder')

    # Data parameters
    parser.add_argument('--data_dir', type=str,
                       default='/DATA1/yunzhu/SSL/training_image/training_image',
                       help='Path to images directory for evaluation')
    parser.add_argument('--image_size', type=int, default=224,
                       help='Size to resize images to')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of samples to evaluate')

    # Feature extraction
    parser.add_argument('--use_cls_token', action='store_true', default=True,
                       help='Use CLS token for features (otherwise use mean of patches)')

    # Output parameters
    parser.add_argument('--output_dir', type=str, default='./mae_evaluation',
                       help='Directory to save evaluation results')
    parser.add_argument('--save_features', action='store_true',
                       help='Save extracted features to disk')
    parser.add_argument('--visualize', action='store_true',
                       help='Create t-SNE and PCA visualizations')
    parser.add_argument('--visualize_attention', action='store_true',
                       help='Visualize attention maps')

    args = parser.parse_args()

    # Print configuration
    print("=" * 80)
    print("MAE Model Evaluation")
    print("=" * 80)
    for arg in vars(args):
        print(f"{arg:25s}: {getattr(args, arg)}")
    print("=" * 80)

    # Run evaluation
    evaluate_model(args)
