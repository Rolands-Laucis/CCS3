import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import umap
import os

def main():
    parser = argparse.ArgumentParser(description='3D UMAP projection of clustered data')
    parser.add_argument('input_file', type=str, help='Path to the input cluster .npy file')
    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"Error: File {args.input_file} not found.")
        return

    print(f"Loading data from {args.input_file}...")
    try:
        # Load the dictionary from the .npy file
        # allow_pickle=True is required because the file contains a dictionary
        data_dict = np.load(args.input_file, allow_pickle=True).item()
        
        if 'data' not in data_dict or 'labels' not in data_dict:
            print("Error: Input file does not contain 'data' or 'labels' keys.")
            return
            
        X = data_dict['data']
        labels = data_dict['labels']
        
        print(f"Data shape: {X.shape}")
        print(f"Labels shape: {labels.shape}")
        
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    print("Running UMAP dimensionality reduction to 3D...")
    reducer = umap.UMAP(n_components=3)
    embedding = reducer.fit_transform(X)
    
    print("Plotting results...")
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2], 
                         c=labels, cmap='viridis', s=3, alpha=0.7)
    
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_zlabel('UMAP 3')
    
    # Add a colorbar
    plt.colorbar(scatter, ax=ax, label='Cluster Label')
    
    plt.title(f'3D UMAP Projection of {os.path.basename(args.input_file)}')
    
    print("Launching interactive 3D window...")
    plt.show()

if __name__ == "__main__":
    main()
