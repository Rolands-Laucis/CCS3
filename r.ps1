./env.ps1
# python dataset.py

# python baseline.py
# python cluster.py --latent_dim 5 --method silhouette --k_max 12 --representations_path "models/baseline" --save_path "results/clusters/baseline"

# python autoencoder.py --latent_dim 16 --hidden_dim 128 --num_epochs 1 --batch_size 32
# python autoencoder.py --latent_dim 16 --hidden_dim 128 --num_epochs 100 --batch_size 8
# python cluster.py --latent_dim 16 --method silhouette --k_max 12 --representations_path "models/autoencoder" --save_path "results/clusters/autoencoder"
# python cluster.py --latent_dim 16 --k 2 --representations_path "models/autoencoder" --save_path "results/clusters/autoencoder"
# python cluster.py --latent_dim 16 --k 3 --representations_path "models/autoencoder" --save_path "results/clusters/autoencoder"
# python cluster.py --latent_dim 16 --k 4 --representations_path "models/autoencoder" --save_path "results/clusters/autoencoder"

# python project_umap.py "results\clusters\autoencoder\clusters_dim16_silhouette_k2.npy"
# python project_umap.py "results\clusters\autoencoder\clusters_dim16_silhouette_k4.npy"
# python project_umap.py "results\clusters\baseline\clusters_dim5_silhouette_k2.npy"

# python evaluation.py --cluster_file "results/clusters/baseline/clusters_dim5_silhouette_k2.npy" 
# python evaluation.py --cluster_file "results/clusters/autoencoder/clusters_dim16_silhouette_k2.npy"

python decoder.py --cluster_path "results/clusters/autoencoder/clusters_dim16_silhouette_k2.npy" --model_path "models\autoencoder\autoencoder_dim16.pth" --metadata_path "models\autoencoder\representations_dim16_metadata.csv"
# python decoder.py --cluster_path "results/clusters/autoencoder/clusters_dim16_silhouette_k3.npy" --model_path "models\autoencoder\autoencoder_dim16.pth" --metadata_path "models\autoencoder\representations_dim16_metadata.csv"
# python decoder.py --cluster_path "results/clusters/autoencoder/clusters_dim16_silhouette_k4.npy" --model_path "models\autoencoder\autoencoder_dim16.pth" --metadata_path "models\autoencoder\representations_dim16_metadata.csv"