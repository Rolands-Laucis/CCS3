import numpy as np
import pandas as pd
from scipy.io import loadmat


def add_saccade_features(x, y, duration):
    """
    Add saccade amplitude and angle as features for each fixation.
    
    Args:
        x, y: Fixation coordinates
        duration: Fixation durations
        
    Returns:
        sequence: Array with shape (num_fixations, 5) containing:
                 [x, y, duration, saccade_amplitude, saccade_angle]
    """
    n_fixations = len(x)
    
    # Initialize saccade features
    saccade_amplitudes = np.zeros(n_fixations)
    saccade_angles = np.zeros(n_fixations)
    
    # For fixations 1 onwards, compute saccade from previous fixation
    if n_fixations > 1:
        # Saccade vectors (from previous to current fixation)
        saccade_x = np.diff(x)  # x[1:] - x[:-1]
        saccade_y = np.diff(y)  # y[1:] - y[:-1]
        
        # Saccade amplitudes (Euclidean distance)
        amplitudes = np.sqrt(saccade_x**2 + saccade_y**2)
        
        # Saccade angles (in degrees, 0° = rightward, 90° = upward)
        angles_rad = np.arctan2(saccade_y, saccade_x)
        angles_deg = np.degrees(angles_rad)
        
        # First fixation has no incoming saccade (0, 0)
        # Subsequent fixations get the saccade that brought us there
        saccade_amplitudes[1:] = amplitudes
        saccade_angles[1:] = angles_deg
    
    # Create sequence with 5 features per timestep
    sequence = np.column_stack([x, y, duration, saccade_amplitudes, saccade_angles])
    
    return sequence


def parse_fixations(mat_file_path):
    """
    Parse MATLAB fixation data into pandas DataFrame and numpy arrays.
    
    Returns:
        df: DataFrame with columns [image, subject_id, fix_x, fix_y, fix_duration, num_fixations]
        sequences: List of dicts with sequences including saccade features
    """
    data = loadmat(mat_file_path)
    fixations = data['fixations']
    
    records = []
    all_sequences = []
    discarded_count = 0
    
    # Iterate over all images (700 images)
    for img_idx in range(fixations.shape[0]):
        img_data = fixations[img_idx, 0][0, 0]
        
        # Extract image name
        img_name = img_data['img'][0]
        
        # Extract subject data
        subjects = img_data['subjects']
        
        # Iterate over all subjects for this image
        for subj_idx in range(subjects.shape[0]):
            subj_data = subjects[subj_idx, 0][0, 0]
            
            # Extract fixation coordinates and durations
            fix_x = subj_data['fix_x'].flatten()
            fix_y = subj_data['fix_y'].flatten()
            fix_duration = subj_data['fix_duration'].flatten()
            
            # Skip sequences with less than 3 fixations
            if len(fix_x) < 3:
                discarded_count += 1
                continue
            
            # Store as record
            record = {
                'image': img_name,
                'image_idx': img_idx,
                'subject_id': subj_idx,
                'fix_x': fix_x,
                'fix_y': fix_y,
                'fix_duration': fix_duration,
                'num_fixations': len(fix_x)
            }
            records.append(record)
            
            # Create temporal sequence with saccade features: (x, y, duration, amplitude, angle)
            sequence = add_saccade_features(fix_x, fix_y, fix_duration)
            all_sequences.append({
                'image': img_name,
                'image_idx': img_idx,
                'subject_id': subj_idx,
                'sequence': sequence  # Shape: (num_fixations, 5)
            })
    
    # Create DataFrame
    df = pd.DataFrame(records)
    
    print(f"Discarded {discarded_count} sequences with < 3 fixations")
    
    return df, all_sequences


def prepare_sequences_for_autoencoder(sequences, max_len=None, padding_value=0.0):
    """
    Prepare padded sequences for autoencoder training.
    
    Args:
        sequences: List of sequence dicts from parse_fixations
        max_len: Maximum sequence length (if None, use max from data)
        padding_value: Value to use for padding
        
    Returns:
        X: Padded sequences array of shape (n_samples, max_len, 5)
        lengths: Original sequence lengths
        metadata: DataFrame with image and subject info
    """
    # Find max sequence length
    seq_lengths = [s['sequence'].shape[0] for s in sequences]
    if max_len is None:
        max_len = max(seq_lengths)
    
    n_samples = len(sequences)
    n_features = 5  # x, y, duration, saccade_amplitude, saccade_angle
    
    # Initialize padded array
    X = np.full((n_samples, max_len, n_features), padding_value, dtype=np.float32)
    
    # Fill in sequences
    for i, s in enumerate(sequences):
        seq = s['sequence']
        seq_len = min(len(seq), max_len)
        X[i, :seq_len, :] = seq[:seq_len]
    
    # Create metadata DataFrame
    metadata = pd.DataFrame([
        {'image': s['image'], 'image_idx': s['image_idx'], 'subject_id': s['subject_id']}
        for s in sequences
    ])
    
    return X, np.array(seq_lengths), metadata


def normalize_sequences(X, method='minmax'):
    """
    Normalize sequences for better autoencoder training.
    
    Args:
        X: Padded sequences array of shape (n_samples, max_len, 5)
        method: 'minmax' or 'zscore'
        
    Returns:
        X_norm: Normalized sequences
        stats: Normalization statistics for inverse transform
    """
    X_norm = X.copy()
    stats = {}
    
    feature_names = ['x', 'y', 'duration', 'saccade_amplitude', 'saccade_angle']
    
    if method == 'minmax':
        for i, feat in enumerate(feature_names):
            feat_min = X[:, :, i][X[:, :, i] != 0].min()
            feat_max = X[:, :, i][X[:, :, i] != 0].max()
            mask = X[:, :, i] != 0
            X_norm[:, :, i] = np.where(mask, (X[:, :, i] - feat_min) / (feat_max - feat_min), 0)
            stats[feat] = {'min': feat_min, 'max': feat_max}
    
    elif method == 'zscore':
        for i, feat in enumerate(feature_names):
            feat_mean = X[:, :, i][X[:, :, i] != 0].mean()
            feat_std = X[:, :, i][X[:, :, i] != 0].std()
            mask = X[:, :, i] != 0
            X_norm[:, :, i] = np.where(mask, (X[:, :, i] - feat_mean) / feat_std, 0)
            stats[feat] = {'mean': feat_mean, 'std': feat_std}
    
    return X_norm, stats


if __name__ == "__main__":
    # Parse the data
    df, sequences = parse_fixations("data/fixations.mat")
    
    print("=" * 60)
    print("PARSED DATA SUMMARY")
    print("=" * 60)
    print(f"Total records (image-subject pairs): {len(df)}")
    print(f"Number of unique images: {df['image'].nunique()}")
    print(f"Subjects per image: {df.groupby('image')['subject_id'].count().unique()}")
    print(f"\nFixation count statistics:")
    print(df['num_fixations'].describe())
    
    print("\n" + "=" * 60)
    print("SAMPLE DATA")
    print("=" * 60)
    print(df.head())
    
    # Prepare for autoencoder
    X, lengths, metadata = prepare_sequences_for_autoencoder(sequences)
    print("\n" + "=" * 60)
    print("AUTOENCODER-READY DATA")
    print("=" * 60)
    print(f"X shape: {X.shape}  (n_samples, max_seq_len, features)")
    print(f"Features: [fix_x, fix_y, fix_duration, saccade_amplitude, saccade_angle]")
    print(f"Sequence length range: {lengths.min()} - {lengths.max()}")
    
    # Normalize
    X_norm, stats = normalize_sequences(X, method='minmax')
    print(f"\nNormalized X range: [{X_norm.min():.3f}, {X_norm.max():.3f}]")
    print(f"Normalization stats: {stats}")
    
    # Example: first sequence
    print("\n" + "=" * 60)
    print("EXAMPLE SEQUENCE (first sample)")
    print("=" * 60)
    print(f"Image: {metadata.iloc[0]['image']}, Subject: {metadata.iloc[0]['subject_id']}")
    print(f"Raw sequence (first 5 fixations):\n{X[0, :5, :]}")
    print(f"Normalized sequence (first 5 fixations):\n{X_norm[0, :5, :]}")