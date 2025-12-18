import argparse
import numpy as np
import matplotlib.pyplot as plt
import umap
import os

def main():
    parser = argparse.ArgumentParser(description='UMAP projection of clustered data')
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

    print("Running UMAP dimensionality reduction...")
    reducer = umap.UMAP(n_components=2)
    embedding = reducer.fit_transform(X)
    
    print("Plotting results...")
    plt.rcParams.update({'font.size': 8})
    plt.figure(figsize=(3, 3))
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='viridis', s=3, alpha=0.7)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Determine save path
    base_name = os.path.splitext(args.input_file)[0]
    save_path = f"{base_name}_umap.png"
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"UMAP plot saved to: {save_path}")

if __name__ == "__main__":
    main()
