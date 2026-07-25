import pandas as pd
import numpy as np
import torch
import os
import joblib
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.recommender_model import SessionRecommenderLSTM

def generate_mixes(query_track_ids, dataset_filepath, model_filepath, scaler_filepath):
    print("--- Starting Playlist Generation Pipeline ---")
    
    print("Loading dataset...")
    df = pd.read_csv(dataset_filepath)
    
    # Extract query tracks
    query_df = df[df['track_id'].isin(query_track_ids)].drop_duplicates(subset=['track_id']).copy()
    
    if query_df.empty:
        print("No valid tracks found for the query.")
        return
        
    output_lines = []
    output_lines.append("--- YOUR RECENTLY PLAYED QUERY ---")
    for _, row in query_df.iterrows():
        output_lines.append(f"- {row['track_name']} by {row['artists']} (Genre: {row['track_genre']})")
    output_lines.append(f"Total Tracks in Query: {len(query_df)}")
    output_lines.append("----------------------------------\n")
    
    feature_cols = [
        "danceability", "energy", "valence", "acousticness", 
        "intensity_index", "groove_index", "euphoria_index", "electronic_index"
    ]
    
    # Load Scaler and Model
    print("Loading Model and Scaler...")
    clustering_data = joblib.load(scaler_filepath)
    scaler = clustering_data["scaler"]
    
    model = SessionRecommenderLSTM(input_dim=len(feature_cols), hidden_dim=32, output_dim=len(feature_cols))
    model.load_state_dict(torch.load(model_filepath, weights_only=True))
    model.eval()
    
    # Prepare query sequence for RNN
    query_features = query_df[feature_cols].fillna(0)
    scaled_query = scaler.transform(query_features)
    
    # Pad or truncate to seq_length of 5 for the general RNN prediction
    seq_length = 5
    if len(scaled_query) < seq_length:
        pad = np.zeros((seq_length - len(scaled_query), len(feature_cols)))
        rnn_input = np.vstack([pad, scaled_query])
    else:
        rnn_input = scaled_query[-seq_length:]
        
    seq_tensor = torch.tensor([rnn_input], dtype=torch.float32)
    
    # Predict overall preference vector
    with torch.no_grad():
        pref_vector = model(seq_tensor).numpy()[0]
        
    # --- DYNAMIC QUERY CLUSTERING USING KMEANS & SILHOUETTE SCORE ---
    print("Applying KMeans to analyze query diversity...")
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    
    max_possible_k = min(6, len(scaled_query) - 1)
    best_k = 1
    best_score = -1
    
    if max_possible_k >= 2:
        for k in range(2, max_possible_k + 1):
            kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=5)
            labels = kmeans_temp.fit_predict(scaled_query)
            # Only consider valid silhouette scores
            if len(set(labels)) > 1:
                score = silhouette_score(scaled_query, labels)
                if score > best_score:
                    best_score = score
                    best_k = k
                    
    # If the score is very low, it implies a single monolithic cluster is better
    if best_score < 0.1 and max_possible_k >= 2:
        best_k = 1
        
    print(f"Optimal K for user query: {best_k} (Silhouette Score: {best_score:.4f})")
    
    kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=5)
    query_clusters = kmeans_final.fit_predict(scaled_query)
    query_df['query_cluster'] = query_clusters
    
    # Generate PCA Plot for visualization
    if len(scaled_query) > 2:
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(scaled_query)
        plt.figure(figsize=(12, 7))
        sns.scatterplot(
            x=pca_result[:, 0], 
            y=pca_result[:, 1], 
            hue=query_clusters, 
            style=query_df['track_genre'],
            palette='viridis', 
            s=150,
            alpha=0.8
        )
        plt.title("KMeans Clustering of User's Query Tracks (PCA Projection)")
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Clusters & Genres")
        plt.tight_layout()
        
        plot_path = os.path.join(os.path.dirname(dataset_filepath), "query_clusters.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved query clustering plot to {plot_path}")
    
    # Calculate similarity across entire dataset for general mix
    print("Finding closest matches in the database...")
    all_features_scaled = scaler.transform(df[feature_cols].fillna(0))
    similarities = cosine_similarity([pref_vector], all_features_scaled)[0]
    
    # Add similarities to df
    df['similarity_score'] = similarities
    candidates = df[~df['track_id'].isin(query_track_ids)].copy()
    
    # --- Generate Mixes ---
    output_lines.append("\n================ GENERATING MIXES ================\n")
    
    # 1. General Top Recommendations (The "For You" Mix)
    for_you = candidates.sort_values(by='similarity_score', ascending=False).head(10)
    output_lines.append("* The 'For You' Mix (Based on your overall session):")
    for _, row in for_you.iterrows():
        output_lines.append(f"  - {row['track_name']} by {row['artists']} (Score: {row['similarity_score']:.2f})")
    output_lines.append("")
    
    # 2. Dynamic Cluster Mixes
    unique_clusters = set(query_clusters)
    num_clusters = len(unique_clusters)
    output_lines.append(f"--- Detected {num_clusters} distinct tastes/vibes in your query ---")
    
    for c_id in sorted(list(unique_clusters)):
        cluster_tracks = query_df[query_df['query_cluster'] == c_id]
        if c_id == -1:
            mix_name = "The 'Eclectic / Diverse' Mix (Noise points)"
        else:
            # Find dominant genre of this cluster to name it
            dominant_genre = cluster_tracks['track_genre'].mode()[0]
            mix_name = f"The '{dominant_genre.title()} & Similar' Mix (Cluster {c_id})"
            
        # Compute sub-preference by averaging features of this cluster
        sub_features = cluster_tracks[feature_cols].fillna(0)
        sub_scaled = scaler.transform(sub_features)
        sub_centroid = np.mean(sub_scaled, axis=0)
        
        # Calculate similarity to this specific centroid
        sub_sims = cosine_similarity([sub_centroid], all_features_scaled)[0]
        # Assign only the subset that corresponds to candidates
        candidates['sub_sim'] = sub_sims[~df['track_id'].isin(query_track_ids)]
        
        cluster_mix = candidates.sort_values(by='sub_sim', ascending=False).head(10)
        
        output_lines.append(f"\n* {mix_name}:")
        for _, row in cluster_mix.iterrows():
            output_lines.append(f"  - {row['track_name']} by {row['artists']} (Genre: {row['track_genre']}, Sub-Score: {row['sub_sim']:.2f})")
    
    output_lines.append("")

    # Save to file
    out_file = os.path.join(os.path.dirname(dataset_filepath), "generated_mixes.txt")
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))

    print(f"--- Playlist Generation Complete. Check {out_file} ---")

if __name__ == "__main__":
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
    
    query_file = os.path.join(PROJECT_ROOT, "query.txt")
    if os.path.exists(query_file):
        with open(query_file, "r") as f:
            query_tracks = [line.strip() for line in f if line.strip()]
    else:
        print(f"No {query_file} found. Creating a default one.")
        df_temp = pd.read_csv(os.path.join(PROJECT_ROOT, "clustered_dataset.csv"))
        query_tracks = df_temp[df_temp['track_genre'] == 'acoustic'].head(5)['track_id'].tolist()
        with open(query_file, "w") as f:
            for t in query_tracks:
                f.write(t + "\n")
    
    generate_mixes(
        query_track_ids=query_tracks,
        dataset_filepath=os.path.join(PROJECT_ROOT, "clustered_dataset.csv"),
        model_filepath=os.path.join(PROJECT_ROOT, "database", "recommender_model.pth"),
        scaler_filepath=os.path.join(PROJECT_ROOT, "database", "clustering_model.pkl")
    )
