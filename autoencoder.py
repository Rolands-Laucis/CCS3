import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import argparse
from dataset import parse_fixations, prepare_sequences_for_autoencoder, normalize_sequences
import os
import csv
from datetime import datetime

class GRUAutoencoder(nn.Module):
    """GRU-based autoencoder for temporal fixation sequences."""
    
    def __init__(self, input_dim=5, hidden_dim=128, latent_dim=16, num_layers=1):
        super(GRUAutoencoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        
        # Encoder
        self.encoder_gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.encoder_fc = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.decoder_fc = nn.Linear(latent_dim, hidden_dim)
        self.decoder_gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.output_fc = nn.Linear(hidden_dim, input_dim)
        
    def encode(self, x):
        """Encode input sequence to latent representation."""
        # x: (batch, seq_len, input_dim)
        _, h_n = self.encoder_gru(x)
        # h_n: (num_layers, batch, hidden_dim) -> take last layer
        h_n = h_n[-1]  # (batch, hidden_dim)
        z = self.encoder_fc(h_n)  # (batch, latent_dim)
        return z
    
    def decode(self, z, seq_len):
        """Decode latent representation back to sequence."""
        # z: (batch, latent_dim)
        h = self.decoder_fc(z)  # (batch, hidden_dim)
        # Repeat for each timestep
        h_repeated = h.unsqueeze(1).repeat(1, seq_len, 1)  # (batch, seq_len, hidden_dim)
        decoded, _ = self.decoder_gru(h_repeated)  # (batch, seq_len, hidden_dim)
        output = self.output_fc(decoded)  # (batch, seq_len, input_dim)
        return output
    
    def forward(self, x):
        """Forward pass through encoder and decoder."""
        seq_len = x.size(1)
        z = self.encode(x)
        reconstructed = self.decode(z, seq_len)
        return reconstructed, z


def create_mask(lengths, max_len, device):
    """Create mask for variable length sequences."""
    batch_size = len(lengths)
    mask = torch.zeros(batch_size, max_len, device=device)
    for i, length in enumerate(lengths):
        mask[i, :length] = 1
    return mask


def masked_mse_loss(pred, target, mask):
    """Compute MSE loss only on non-padded positions."""
    mask = mask.unsqueeze(-1)  # (batch, seq_len, 1)
    loss = ((pred - target) ** 2) * mask
    return loss.sum() / mask.sum()


def train_autoencoder(model, train_loader, lengths_array, num_epochs=100, lr=1e-3, device='cuda'):
    """Train the autoencoder model."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    losses = []
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch_idx, (data, lengths) in enumerate(train_loader):
            data = data.to(device)
            lengths = lengths.to(device)
            
            optimizer.zero_grad()
            
            reconstructed, z = model(data)
            
            # Create mask for variable length sequences
            mask = create_mask(lengths, data.size(1), device)
            loss = masked_mse_loss(reconstructed, data, mask)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}")
    
    return losses


def extract_representations(model, data_loader, device='cuda'):
    """Extract latent representations for all sequences."""
    model.eval()
    representations = []
    
    with torch.no_grad():
        for data, _ in data_loader:
            data = data.to(device)
            z = model.encode(data)
            representations.append(z.cpu().numpy())
    
    return np.vstack(representations)


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train GRU Autoencoder for fixation sequences')
    parser.add_argument('--latent_dim', type=int, default=30,
                        help='Dimension of latent representation (default: 30)')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Hidden dimension of GRU (default: 128)')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of training epochs (default: 100)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size (default: 64)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--data_path', type=str, default='data/fixations.mat',
                        help='Path to fixations.mat file')
    parser.add_argument('--save_path', type=str, default='models/autoencoder',
                        help='Path to save the model and representations')
    args = parser.parse_args()

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load and prepare data
    print("\nLoading data...")
    df, sequences = parse_fixations(args.data_path)
    X, lengths, metadata = prepare_sequences_for_autoencoder(sequences)
    X_norm, stats = normalize_sequences(X, method='minmax')
    
    print(f"Data shape: {X_norm.shape}")
    print(f"Number of sequences: {len(X_norm)}")
    
    # Convert to PyTorch tensors
    X_tensor = torch.FloatTensor(X_norm)
    lengths_tensor = torch.LongTensor(lengths)
    
    # Create DataLoader
    dataset = TensorDataset(X_tensor, lengths_tensor)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    full_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    # Train autoencoder
    print("\n" + "=" * 60)
    print(f"Training Autoencoder with latent_dim = {args.latent_dim}")
    print("=" * 60)
    
    model = GRUAutoencoder(
        input_dim=5,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        num_layers=1
    )
    
    losses = train_autoencoder(
        model, 
        train_loader, 
        lengths, 
        num_epochs=args.num_epochs, 
        lr=args.lr, 
        device=device
    )
    
    # Extract representations
    representations = extract_representations(model, full_loader, device)
    
    # Summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Final loss: {losses[-1]:.6f}")
    print(f"Representations shape: {representations.shape}")
    
    # Save model and representations
    print("\nSaving model and representations...")
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    model_save_path = os.path.join(args.save_path, f"autoencoder_dim{args.latent_dim}.pth")
    torch.save(model.state_dict(), model_save_path)
    np.save(os.path.join(args.save_path, f"representations_dim{args.latent_dim}.npy"), representations)
    
    # Save metadata
    metadata_path = os.path.join(args.save_path, f"representations_dim{args.latent_dim}_metadata.csv")
    metadata.to_csv(metadata_path, index=False)
    print(f"Metadata saved to {metadata_path}")
    
    # Log run to CSV
    csv_path = os.path.join(args.save_path, "autoencoder_experimental_runs.csv")
    csv_exists = os.path.exists(csv_path)
    
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not csv_exists:
            writer.writerow(['timestamp', 'latent_dim', 'hidden_dim', 'num_epochs', 'batch_size', 'final_loss'])
        writer.writerow([
            datetime.now().timestamp(),
            args.latent_dim,
            args.hidden_dim,
            args.num_epochs,
            args.batch_size,
            f"{losses[-1]:.6f}"
        ])
    print(f"Run logged to {csv_path}")
    
    print("Done!")