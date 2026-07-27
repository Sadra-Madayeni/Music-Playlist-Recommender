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
    
    # Extract query tracks PRESERVING exact chronological order and allowing repetitions
    df_indexed = df.set_index('track_id')
    valid_ids = [tid for tid in query_track_ids if tid in df_indexed.index]
    query_df = df_indexed.loc[valid_ids].reset_index().copy()
    
    if query_df.empty:
        print("No valid tracks found for the query.")
        return
        
    output_lines = []
    output_lines.append("--- YOUR RECENTLY PLAYED QUERY ---")
    for _, row in query_df.iterrows():
        output_lines.append(f"- {row['track_name']} by {row['artists']} (Genre: {row['track_genre']})")
    # Count artists and genres from the query for personalized weighting
    from collections import Counter
    all_query_artists = []
    for artists_str in query_df['artists'].dropna():
        all_query_artists.extend([x.strip() for x in artists_str.split(';')])
    artist_counts = Counter(all_query_artists)
    
    query_genre_counts = query_df['track_genre'].value_counts().to_dict()
    
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
        
    # --- METADATA-DRIVEN QUERY CLUSTERING USING KMEANS ---
    print("Applying PCA and metadata-driven KMeans to analyze query diversity...")
    from sklearn.cluster import KMeans
    
    # Run PCA first so clustering matches visual density
    pca_result = None
    if len(scaled_query) > 2:
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(scaled_query)
        clustering_input = pca_result
    else:
        clustering_input = scaled_query
        
    # Analyze cultural metadata to determine exact K
    genre_counts = query_df['track_genre'].value_counts()
    
    # Define a "main genre" as any genre that makes up at least 5% of the query (minimum 3 tracks)
    min_tracks = max(3, int(len(query_df) * 0.05))
    main_genres = genre_counts[genre_counts >= min_tracks]
    
    # Set K explicitly to the number of main genres found!
    dynamic_k = len(main_genres)
    
    # Bound K to avoid fragmentation or breaking KMeans
    best_k = max(1, min(dynamic_k, 6, len(scaled_query) - 1))
    
    print(f"Detected {len(main_genres)} main genres (>= {min_tracks} tracks). Setting K={best_k}.")
    
    kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=5)
    query_clusters = kmeans_final.fit_predict(clustering_input)
        
    query_df['query_cluster'] = query_clusters
    
    # Generate PCA Plot for visualization
    if pca_result is not None:
        plt.figure(figsize=(12, 7))
        
        sns.scatterplot(
            x=pca_result[:, 0], 
            y=pca_result[:, 1], 
            hue=query_df['track_genre'], 
            style=query_clusters,
            s=150,
            alpha=0.8
        )
        plt.title(f"Metadata-Driven KMeans Clustering (K={best_k})")
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
    
    # Add similarities to df BEFORE creating candidates
    df['similarity_score'] = similarities
    
    # Filter candidates to exclude songs the user just played (by name and artist to catch alternate track_ids)
    merged = df.merge(query_df[['track_name', 'artists']].drop_duplicates(), on=['track_name', 'artists'], how='left', indicator=True)
    candidates = df[merged['_merge'] == 'left_only'].copy()
    
    # --- Generate Mixes ---
    output_lines.append("\n================ GENERATING MIXES ================\n")
    
    # Calculate artist bonus
    def get_artist_bonus(artists_str):
        if pd.isna(artists_str): return 0.0
        bonus = 0.0
        for a in artists_str.split(';'):
            # 0.005 bonus per listen, max 0.05 per artist
            bonus += min(artist_counts.get(a.strip(), 0) * 0.005, 0.05)
        return bonus
    
    candidates['artist_bonus'] = candidates['artists'].apply(get_artist_bonus)
    
    # 1. General Top Recommendations (The "For You" Mix)
    def get_for_you_genre_bonus(g_name):
        if pd.isna(g_name): return -0.10
        count = query_genre_counts.get(g_name, 0)
        if count == 0:
            return -0.10 # Heavy penalty for random genres not in query
        else:
            return min(count * 0.005, 0.05) # Boost slightly if they did listen to it
            
    candidates['genre_bonus'] = candidates['track_genre'].apply(get_for_you_genre_bonus)
    candidates['for_you_score'] = candidates['similarity_score'] + candidates['artist_bonus'] + candidates['genre_bonus']
    for_you = candidates.sort_values(by='for_you_score', ascending=False).drop_duplicates(subset=['track_name', 'artists']).head(10)
    output_lines.append("* The 'For You' Mix (Based on your overall session):")
    for _, row in for_you.iterrows():
        output_lines.append(f"  - {row['track_name']} by {row['artists']} (Genre: {row['track_genre']}, Score: {row['for_you_score']:.2f}, Artist Bonus: {row['artist_bonus']:.3f})")
    output_lines.append("")
    
    # 2. Dynamic Cluster Mixes
    unique_clusters = set(query_clusters)
    
    genre_mixes = {}
    
    for c_id in sorted(list(unique_clusters)):
        if c_id == -1: continue
        cluster_tracks = query_df[query_df['query_cluster'] == c_id]
        dominant_genre = cluster_tracks['track_genre'].mode()[0]
            
        # Use LSTM to predict the next track's features for this specific vibe
        sub_features = cluster_tracks[feature_cols].fillna(0)
        sub_scaled = scaler.transform(sub_features)
        
        if len(sub_scaled) < seq_length:
            pad = np.zeros((seq_length - len(sub_scaled), len(feature_cols)))
            cluster_rnn_input = np.vstack([pad, sub_scaled])
        else:
            cluster_rnn_input = sub_scaled[-seq_length:]
            
        cluster_tensor = torch.tensor([cluster_rnn_input], dtype=torch.float32)
        with torch.no_grad():
            sub_centroid = model(cluster_tensor).numpy()[0]
        
        # Calculate similarity to this specific centroid
        sub_sims = cosine_similarity([sub_centroid], all_features_scaled)[0]
        
        # Assign to candidates safely using their retained original index
        candidates['sub_sim'] = sub_sims[candidates.index]
        
        if 'similar_genre_1' in df.columns:
            dom_row = cluster_tracks[cluster_tracks['track_genre'] == dominant_genre].iloc[0]
            
            def get_genre_weight(g_name, base_w):
                if pd.isna(g_name): return 0.0
                count = query_genre_counts.get(g_name, 0)
                if count == 0:
                    return base_w - 0.15 # Penalize heavily if the user didn't listen to this genre at all
                else:
                    return base_w + min(count * 0.005, 0.02) # Boost slightly if they did
            
            allowed = {
                dominant_genre: 1.0,
                dom_row.get('similar_genre_1'): get_genre_weight(dom_row.get('similar_genre_1'), 0.985),
                dom_row.get('similar_genre_2'): get_genre_weight(dom_row.get('similar_genre_2'), 0.980),
                dom_row.get('similar_genre_3'): get_genre_weight(dom_row.get('similar_genre_3'), 0.975)
            }
            allowed = {k: v for k, v in allowed.items() if pd.notna(k)}
            
            candidates['weight'] = candidates['track_genre'].map(allowed).fillna(0.0)
            candidates['weighted_sim'] = (candidates['sub_sim'] * candidates['weight']) + candidates['artist_bonus']
            valid_candidates = candidates[candidates['weight'] > 0]
            cluster_mix = valid_candidates.sort_values(by='weighted_sim', ascending=False).drop_duplicates(subset=['track_name', 'artists']).head(10)
        else:
            candidates['weighted_sim'] = candidates['sub_sim'] + candidates['artist_bonus']
            cluster_mix = candidates.sort_values(by='weighted_sim', ascending=False).drop_duplicates(subset=['track_name', 'artists']).head(10)
            
        if dominant_genre not in genre_mixes:
            genre_mixes[dominant_genre] = []
        genre_mixes[dominant_genre].append(cluster_mix)
        
    # Combine mixes by genre
    actual_num_vibes = len(genre_mixes)
    output_lines.append(f"\n--- Detected {actual_num_vibes} distinct tastes/vibes in your query ---")
    for dominant_genre, mix_list in genre_mixes.items():
        combined_mix = pd.concat(mix_list)
        final_mix = combined_mix.sort_values(by='weighted_sim', ascending=False).drop_duplicates(subset=['track_name', 'artists']).head(10)
        
        mix_name = f"The '{dominant_genre.title()} & Similar' Mix"
        output_lines.append(f"\n* {mix_name}:")
        for _, row in final_mix.iterrows():
            output_lines.append(f"  - {row['track_name']} by {row['artists']} (Genre: {row['track_genre']}, Sub-Score: {row['weighted_sim']:.2f}, Artist Bonus: {row['artist_bonus']:.3f})")
    
    output_lines.append("")
    
    # 3. Dynamic Artist Mixes (For artists with 5+ plays)
    output_lines.append(f"--- Dedicated Artist Mixes ---")
    artist_mix_generated = False
    
    for artist, count in artist_counts.items():
        if count >= 5:
            # Get user's played tracks for this artist
            artist_query_tracks = query_df[query_df['artists'].str.contains(artist, na=False, case=False, regex=False)]
            if len(artist_query_tracks) == 0: continue
                
            # Use LSTM to predict the next track's features for this specific artist
            artist_features = artist_query_tracks[feature_cols].fillna(0)
            artist_scaled = scaler.transform(artist_features)
            
            if len(artist_scaled) < seq_length:
                pad = np.zeros((seq_length - len(artist_scaled), len(feature_cols)))
                artist_rnn_input = np.vstack([pad, artist_scaled])
            else:
                artist_rnn_input = artist_scaled[-seq_length:]
                
            artist_tensor = torch.tensor([artist_rnn_input], dtype=torch.float32)
            with torch.no_grad():
                artist_centroid = model(artist_tensor).numpy()[0]
            
            # Find similarity to all tracks
            artist_sims = cosine_similarity([artist_centroid], all_features_scaled)[0]
            
            # Filter candidates to ONLY this artist
            artist_candidates = df[~df['track_id'].isin(query_track_ids) & df['artists'].str.contains(artist, na=False, case=False, regex=False)].copy()
            if len(artist_candidates) == 0: continue
            
            artist_candidates['sim'] = artist_sims[artist_candidates.index]
            artist_mix = artist_candidates.sort_values(by='sim', ascending=False).drop_duplicates(subset=['track_name', 'artists']).head(10)
            
            if len(artist_mix) > 0:
                artist_mix_generated = True
                mix_name = f"The '{artist}' Mix"
                output_lines.append(f"\n* {mix_name}:")
                for _, row in artist_mix.iterrows():
                    output_lines.append(f"  - {row['track_name']} by {row['artists']} (Genre: {row['track_genre']}, Sub-Score: {row['sim']:.2f})")
    
    if not artist_mix_generated:
        output_lines.append("\n(No single artist had enough plays to generate a dedicated mix.)")
        
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
