import numpy as np
import argparse
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import os


def find_optimal_k_elbow(X, k_range, random_state=42):
    """Find optimal k using elbow method (inertia)."""
    inertias = []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
    
    # Find elbow point using second derivative
    if len(k_range) > 2:
        diffs = np.diff(inertias)
        second_diffs = np.diff(diffs)
        elbow_idx = np.argmax(second_diffs) + 2  # +2 because of two diffs
        optimal_k = k_range[elbow_idx]
    else:
        optimal_k = k_range[0]
    
    return optimal_k, inertias


def find_optimal_k_silhouette(X, k_range, random_state=42):
    """Find optimal k using silhouette score."""
    silhouette_scores = []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(X)
        score = silhouette_score(X, labels)
        silhouette_scores.append(score)
    
    # Find k with highest silhouette score
    optimal_idx = np.argmax(silhouette_scores)
    optimal_k = k_range[optimal_idx]
    
    return optimal_k, silhouette_scores


def plot_elbow(k_range, inertias, optimal_k, save_path=None):
    """Plot elbow curve."""
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
    plt.axvline(x=optimal_k, color='r', linestyle='--', label=f'Optimal k={optimal_k}')
    plt.xlabel('Number of Clusters (k)', fontsize=12)
    plt.ylabel('Inertia', fontsize=12)
    plt.title('Elbow Method for Optimal k', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Elbow plot saved to: {save_path}")
    plt.close()


def plot_silhouette(k_range, scores, optimal_k, save_path=None):
    """Plot silhouette scores."""
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, scores, 'go-', linewidth=2, markersize=8)
    plt.axvline(x=optimal_k, color='r', linestyle='--', label=f'Optimal k={optimal_k}')
    plt.xlabel('Number of Clusters (k)', fontsize=12)
    plt.ylabel('Silhouette Score', fontsize=12)
    plt.title('Silhouette Method for Optimal k', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Silhouette plot saved to: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Cluster latent representations using K-means')
    parser.add_argument('--latent_dim', type=int, default=10,
                        help='Dimension of latent representations (default: 10)')
    parser.add_argument('--method', type=str, choices=['silhouette', 'elbow'], default='silhouette',
                        help='Method to find optimal k (default: silhouette)')
    parser.add_argument('--k', type=int, default=None,
                        help='Override: manually set number of clusters')
    parser.add_argument('--k_min', type=int, default=2,
                        help='Minimum k to try (default: 2)')
    parser.add_argument('--k_max', type=int, default=15,
                        help='Maximum k to try (default: 15)')
    parser.add_argument('--representations_path', type=str, default='models/autoencoder',
                        help='Path to representations directory')
    parser.add_argument('--save_path', type=str, default='results/clusters',
                        help='Path to save clustering results')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random state for reproducibility')
    args = parser.parse_args()
    
    # Load representations
    rep_file = f"{args.representations_path}/representations_dim{args.latent_dim}.npy"
    print(f"Loading representations from: {rep_file}")
    
    if not os.path.exists(rep_file):
        raise FileNotFoundError(f"Representations file not found: {rep_file}")
    
    X = np.load(rep_file)
    print(f"Loaded representations shape: {X.shape}")
    
    # Create save directory
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    
    # Determine optimal k
    k_range = list(range(args.k_min, args.k_max + 1))
    
    if args.k is not None:
        # Manual override
        optimal_k = args.k
        print(f"\nUsing manually specified k = {optimal_k}")
    else:
        print(f"\nFinding optimal k using {args.method} method...")
        print(f"Testing k in range [{args.k_min}, {args.k_max}]")
        
        if args.method == 'silhouette':
            optimal_k, scores = find_optimal_k_silhouette(X, k_range, args.random_state)
            plot_silhouette(k_range, scores, optimal_k, 
                           f"{args.save_path}/silhouette_dim{args.latent_dim}.png")
            print(f"\nSilhouette scores: {dict(zip(k_range, [f'{s:.4f}' for s in scores]))}")
        else:
            optimal_k, inertias = find_optimal_k_elbow(X, k_range, args.random_state)
            plot_elbow(k_range, inertias, optimal_k,
                      f"{args.save_path}/elbow_dim{args.latent_dim}.png")
            print(f"\nInertias: {dict(zip(k_range, [f'{i:.2f}' for i in inertias]))}")
        
        print(f"\nOptimal k found: {optimal_k}")
    
    # Perform final clustering
    print(f"\nRunning K-means with k = {optimal_k}...")
    kmeans = KMeans(n_clusters=optimal_k, random_state=args.random_state, n_init=10)
    labels = kmeans.fit_predict(X)
    
    # Compute final metrics
    final_silhouette = silhouette_score(X, labels)
    
    print("\n" + "=" * 60)
    print("CLUSTERING RESULTS")
    print("=" * 60)
    print(f"Number of clusters: {optimal_k}")
    print(f"Silhouette score: {final_silhouette:.4f}")
    print(f"Inertia: {kmeans.inertia_:.4f}")
    print(f"\nCluster distribution:")
    unique, counts = np.unique(labels, return_counts=True)
    for cluster, count in zip(unique, counts):
        print(f"  Cluster {cluster}: {count} samples ({100*count/len(labels):.1f}%)")
    
    # Save results
    np.save(f"{args.save_path}/labels_dim{args.latent_dim}_{args.method}_k{optimal_k}.npy", labels)
    np.save(f"{args.save_path}/centroids_dim{args.latent_dim}_{args.method}_k{optimal_k}.npy", kmeans.cluster_centers_)
    
    print(f"\nResults saved to: {args.save_path}")
    print(f"  - labels_dim{args.latent_dim}_{args.method}_k{optimal_k}.npy")
    print(f"  - centroids_dim{args.latent_dim}_{args.method}_k{optimal_k}.npy")


if __name__ == "__main__":
    main()
