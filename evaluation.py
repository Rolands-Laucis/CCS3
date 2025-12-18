import numpy as np
import argparse
import os
import csv
from datetime import datetime
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

def main():
    parser = argparse.ArgumentParser(description='Evaluate clustering results')
    parser.add_argument('--cluster_file', type=str, required=True,
                        help='Path to the cluster .npy file containing dictionary with keys: data, centroids, labels')
    parser.add_argument('--results_dir', type=str, default='results/eval',
                        help='Directory to save evaluation results')
    
    args = parser.parse_args()

    # Load data
    if not os.path.exists(args.cluster_file):
        raise FileNotFoundError(f"Cluster file not found: {args.cluster_file}")
    
    print(f"Loading cluster data from {args.cluster_file}...")
    try:
        # Load the dictionary (allow_pickle=True is required for object arrays/dicts)
        cluster_data = np.load(args.cluster_file, allow_pickle=True).item()
    except Exception as e:
        raise ValueError(f"Failed to load cluster file: {e}")

    if not isinstance(cluster_data, dict):
        raise ValueError(f"File {args.cluster_file} does not contain a dictionary.")

    # Extract components
    try:
        X = cluster_data['data']
        labels = cluster_data['labels']
        centroids = cluster_data['centroids']
    except KeyError as e:
        raise KeyError(f"Cluster file missing required key: {e}")

    print(f"Data shape: {X.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Centroids shape: {centroids.shape}")

    # Check consistency
    if len(X) != len(labels):
        raise ValueError(f"Data and labels have different number of samples: {len(X)} vs {len(labels)}")

    # Compute metrics
    print("\nComputing metrics...")
    
    # Silhouette Score
    print("Computing Silhouette Score...")
    sil_score = silhouette_score(X, labels)
    print(f"Silhouette Score: {sil_score:.4f}")

    # Davies-Bouldin Index
    print("Computing Davies-Bouldin Index...")
    db_score = davies_bouldin_score(X, labels)
    print(f"Davies-Bouldin Index: {db_score:.4f}")

    # Calinski-Harabasz Index
    print("Computing Calinski-Harabasz Index...")
    ch_score = calinski_harabasz_score(X, labels)
    print(f"Calinski-Harabasz Index: {ch_score:.4f}")

    # Save results to CSV
    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)
    
    csv_file = os.path.join(args.results_dir, 'clustering_evaluation.csv')
    file_exists = os.path.isfile(csv_file)
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    results = {
        'timestamp': timestamp,
        'cluster_file': args.cluster_file,
        'n_clusters': len(np.unique(labels)),
        'silhouette_score': sil_score,
        'davies_bouldin_score': db_score,
        'calinski_harabasz_score': ch_score
    }
    
    fieldnames = ['timestamp', 'cluster_file', 'n_clusters', 
                  'silhouette_score', 'davies_bouldin_score', 'calinski_harabasz_score']
    
    with open(csv_file, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(results)
        
    print(f"\nResults appended to {csv_file}")

if __name__ == "__main__":
    main()
