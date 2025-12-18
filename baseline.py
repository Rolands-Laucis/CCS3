import numpy as np
import pandas as pd
import os
from dataset import parse_fixations

def create_baseline_representations():
    # Ensure output directory exists
    output_dir = os.path.join('models', 'baseline')
    os.makedirs(output_dir, exist_ok=True)
    
    # Parse the data
    print("Loading and parsing fixations...")
    _, sequences = parse_fixations("data/fixations.mat")
    
    baseline_vectors = []
    metadata_list = []
    
    print(f"Processing {len(sequences)} sequences...")
    
    for seq_data in sequences:
        sequence = seq_data['sequence'] # Shape (n_fixations, 5)
        
        # Features: [x, y, duration, saccade_amplitude, saccade_angle]
        feature_means = []
        
        for i in range(5):
            feature_values = sequence[:, i]
            # Exclude 0 values
            non_zero_values = feature_values[feature_values != 0]
            
            if len(non_zero_values) > 0:
                mean_val = np.mean(non_zero_values)
            else:
                mean_val = 0.0
                
            feature_means.append(mean_val)
            
        baseline_vectors.append(feature_means)
        
        metadata_list.append({
            'image': seq_data['image'],
            'image_idx': seq_data['image_idx'],
            'subject_id': seq_data['subject_id']
        })
        
    # Convert to numpy array
    X = np.array(baseline_vectors, dtype=np.float32)
    
    # Save representations
    npy_path = os.path.join(output_dir, 'representations_dim5.npy')
    np.save(npy_path, X)
    print(f"Saved representations to {npy_path}")
    print(f"Shape: {X.shape}")
    
    # Save metadata
    metadata_df = pd.DataFrame(metadata_list)
    csv_path = os.path.join(output_dir, 'representations_dim5_metadata.csv')
    metadata_df.to_csv(csv_path, index=False)
    print(f"Saved metadata to {csv_path}")

if __name__ == "__main__":
    create_baseline_representations()
