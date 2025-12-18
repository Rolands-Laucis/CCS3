import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import colorsys
import pandas as pd
import os
from PIL import Image
from autoencoder import GRUAutoencoder
from dataset import parse_fixations

def load_model(model_path, latent_dim=16, hidden_dim=128, device='cpu'):
    """Load the trained autoencoder model."""
    model = GRUAutoencoder(input_dim=5, hidden_dim=hidden_dim, latent_dim=latent_dim)
    # Load weights
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model.to(device)
    model.eval()
    return model

def decode_vector(model, z, seq_len, device='cpu'):
    """Decode a single latent vector z into a sequence."""
    with torch.no_grad():
        # z shape: (latent_dim,) -> (1, latent_dim)
        z_tensor = torch.FloatTensor(z).unsqueeze(0).to(device)
        # decoded shape: (1, seq_len, 5)
        decoded = model.decode(z_tensor, seq_len)
        return decoded.cpu().numpy()[0]

def create_gradient_colors(base_color, n_points, hue_shift=0.15):
    """Generate a gradient of colors by shifting hue."""
    rgb = mcolors.to_rgb(base_color)
    h, l, s = colorsys.rgb_to_hls(*rgb)
    
    colors = []
    for i in range(n_points):
        # Shift hue slightly
        dh = (i / max(1, n_points - 1)) * hue_shift
        new_h = (h + dh) % 1.0
        new_rgb = colorsys.hls_to_rgb(new_h, l, s)
        colors.append(new_rgb)
    return colors

def visualize_scanpath(image_path, scanpaths, save_path):
    """
    Visualize scanpaths on the image.
    
    Args:
        image_path: Path to the background image
        scanpaths: List of tuples (sequence, color, label, alpha, linewidth)
                   sequence is (seq_len, 5) array where cols 0,1 are x,y
        save_path: Where to save the plot
    """
    try:
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}. Creating blank canvas.")
            # Create a blank white image if file not found
            img = Image.new('RGB', (1024, 768), color='white')
        else:
            img = Image.open(image_path)
        
        width, height = img.size
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return

    plt.figure(figsize=(3, 3))
    plt.imshow(img)
    
    for seq, base_color, label, _, lw in scanpaths:
        # Override alpha
        alpha = 0.5
        
        # Denormalize x, y
        # Assuming normalized to [0, 1]
        # Clip to [0, 1] just in case
        x_norm = np.clip(seq[:, 0], 0, 1)
        y_norm = np.clip(seq[:, 1], 0, 1)
        
        x = x_norm * width
        y = y_norm * height
        
        # Generate gradient colors
        n_points = len(x)
        colors = create_gradient_colors(base_color, n_points, hue_shift=0.1)
        
        # Plot segments
        for i in range(n_points - 1):
            plt.plot(x[i:i+2], y[i:i+2], color=colors[i], linewidth=lw, alpha=alpha)
            
        # Plot points (smaller size)
        plt.scatter(x, y, c=colors, s=5, alpha=alpha, zorder=10)
        
        # Mark start (triangle) - smaller
        plt.plot(x[0], y[0], color=colors[0], marker='^', markersize=10, markeredgecolor='white', alpha=alpha, zorder=11)
        # Mark end (square) - smaller
        plt.plot(x[-1], y[-1], color=colors[-1], marker='s', markersize=10, markeredgecolor='white', alpha=alpha, zorder=11)
        
        # Dummy plot for legend
        plt.plot([], [], color=base_color, linewidth=lw, label=label, alpha=alpha)

    # Only show unique labels in legend (smaller and tighter)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), fontsize=8, framealpha=1)
    
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Saved visualization to {save_path}")

def main():
    parser = argparse.ArgumentParser(description='Decode and visualize cluster centroids')
    parser.add_argument('--cluster_path', type=str, required=True, 
                        help='Path to clusters .npy file')
    parser.add_argument('--model_path', type=str, default='models/autoencoder/autoencoder_dim16.pth', 
                        help='Path to model .pth file')
    parser.add_argument('--metadata_path', type=str, default='models/autoencoder/representations_dim16_metadata.csv', 
                        help='Path to metadata .csv file')
    parser.add_argument('--image_dir', type=str, default='data/image_stimuli', 
                        help='Directory containing images')
    parser.add_argument('--save_dir', type=str, default='results/decode', 
                        help='Directory to save visualizations')
    parser.add_argument('--seq_len', type=int, default=22, 
                        help='Sequence length for decoding (default: 22)')
    parser.add_argument('--n_neighbors', type=int, default=10, 
                        help='Number of neighbors to visualize')
    parser.add_argument('--fixations_path', type=str, default='data/fixations.mat',
                        help='Path to fixations.mat file')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 0. Load Original Fixations (for visualization)
    print(f"Loading original fixations from {args.fixations_path}")
    if os.path.exists(args.fixations_path):
        _, all_sequences = parse_fixations(args.fixations_path)
        # Organize by image for fast lookup
        image_scanpaths = {}
        for seq_data in all_sequences:
            img = seq_data['image']
            if img not in image_scanpaths:
                image_scanpaths[img] = []
            image_scanpaths[img].append(seq_data)
        print(f"Loaded original scanpaths for {len(image_scanpaths)} images")
    else:
        print(f"Warning: Fixations file not found at {args.fixations_path}. Original scanpaths will not be shown.")
        image_scanpaths = {}
    
    # 1. Load Cluster Data
    print(f"Loading clusters from {args.cluster_path}")
    if not os.path.exists(args.cluster_path):
        raise FileNotFoundError(f"Cluster file not found: {args.cluster_path}")
        
    cluster_data = np.load(args.cluster_path, allow_pickle=True).item()
    # Keys: 'data', 'centroids', 'labels'
    
    latent_data = cluster_data['data']
    centroids = cluster_data['centroids']
    labels = cluster_data['labels']
    
    print(f"Loaded data shape: {latent_data.shape}")
    print(f"Loaded centroids shape: {centroids.shape}")
    
    # 2. Load Metadata
    print(f"Loading metadata from {args.metadata_path}")
    if not os.path.exists(args.metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {args.metadata_path}")
        
    metadata = pd.read_csv(args.metadata_path)
    
    # Verify alignment
    if len(metadata) != len(latent_data):
        print(f"WARNING: Metadata length ({len(metadata)}) does not match data length ({len(latent_data)})")
    
    # 3. Load Model
    print(f"Loading model from {args.model_path}")
    # Infer latent_dim from centroids shape (k, latent_dim)
    latent_dim = centroids.shape[1]
    print(f"Inferred latent_dim: {latent_dim}")
    
    model = load_model(args.model_path, latent_dim=latent_dim, device=device)
    
    # 4. Create Save Directory
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        
    # 5. Process Each Cluster
    unique_clusters = np.unique(labels)
    print(f"Found {len(unique_clusters)} clusters: {unique_clusters}")
    
    for cluster_id in unique_clusters:
        print(f"\nProcessing Cluster {cluster_id}")
        
        # Get centroid vector
        centroid = centroids[cluster_id]
        
        # Get data points belonging to this cluster
        cluster_indices = np.where(labels == cluster_id)[0]
        cluster_points = latent_data[cluster_indices]
        
        if len(cluster_points) == 0:
            print(f"Skipping cluster {cluster_id} (no points)")
            continue
            
        # Find closest neighbors to centroid within the cluster
        # Calculate Euclidean distances
        distances = np.linalg.norm(cluster_points - centroid, axis=1)
        
        # Get indices of n closest neighbors
        n_vis = min(args.n_neighbors, len(cluster_points))
        closest_local_indices = np.argsort(distances)[:n_vis]
        closest_global_indices = cluster_indices[closest_local_indices]

        # Decode centroid
        decoded_centroid = decode_vector(model, centroid, args.seq_len, device)
        
        # Iterate over the nearest neighbors to generate visualizations
        for rank, global_idx in enumerate(closest_global_indices):
            # Get metadata for this specific point
            if global_idx < len(metadata):
                meta_row = metadata.iloc[global_idx]
                image_name = meta_row['image']
                # Try to get subject_id, default to None if not found
                subject_id = meta_row.get('subject_id', None) 
                if subject_id is None:
                     # Fallback for column naming variations
                     subject_id = meta_row.get('subject', None)
            else:
                print(f"Index {global_idx} out of bounds for metadata.")
                continue

            print(f"  Neighbor {rank+1}: Image={image_name}, Subject={subject_id}")
            
            image_path = os.path.join(args.image_dir, image_name)
            
            scanpaths_to_plot = []
            
            # 1. Add Original Scanpath for this specific subject (Green)
            if image_name in image_scanpaths:
                # Find the specific subject's sequence
                found_subject = False
                if subject_id is not None:
                    for seq_data in image_scanpaths[image_name]:
                        # Compare subject_ids (handle potential type mismatch str vs int)
                        if str(seq_data['subject_id']) == str(subject_id):
                            orig_seq = seq_data['sequence']
                            
                            # Normalize
                            try:
                                with Image.open(image_path) as img:
                                    w, h = img.size
                                norm_seq = orig_seq.copy()
                                norm_seq[:, 0] = norm_seq[:, 0] / w
                                norm_seq[:, 1] = norm_seq[:, 1] / h
                                
                                scanpaths_to_plot.append((norm_seq, 'lime', f'Participant {subject_id}', 0.8, 1.5))
                                found_subject = True
                                break
                            except Exception as e:
                                print(f"    Failed to load image for normalization: {e}")
                
                if not found_subject:
                    print(f"    Warning: Original scanpath for subject {subject_id} on {image_name} not found in fixations.")

            # 2. Add centroid (Red, thick, on top)
            scanpaths_to_plot.append((decoded_centroid, 'red', f'centroid {cluster_id + 1}', 1.0, 2.0))
            
            # Visualize
            image_base_name = os.path.splitext(image_name)[0]
            # Include rank in filename to distinguish multiple neighbors
            save_name = f"cluster_dim{latent_dim}_k{cluster_id}_n{rank+1}_{image_base_name}.png"
            save_path = os.path.join(args.save_dir, save_name)
            
            visualize_scanpath(image_path, scanpaths_to_plot, save_path)

    print("\nDone!")

if __name__ == "__main__":
    main()
