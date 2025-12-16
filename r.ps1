./env.ps1
python dataset.py
# python autoencoder.py --latent_dim 10 --hidden_dim 64 --num_epochs 10 --batch_size 16
# python cluster.py --latent_dim 10 --method silhouette --k_max 30
# python cluster.py --latent_dim 10 --method elbow

# python autoencoder.py --latent_dim 8 --hidden_dim 64 --num_epochs 10 --batch_size 64
# python cluster.py --latent_dim 8 --method silhouette --k_max 20