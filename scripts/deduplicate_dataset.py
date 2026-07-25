import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import os

def deduplicate_dataset(filepath):
    print("Loading dataset...")
    df = pd.read_csv(filepath)
    original_len = len(df)
    
    feature_cols = [
        "danceability", "energy", "valence", "acousticness", 
        "intensity_index", "groove_index", "euphoria_index", "electronic_index"
    ]
    
    print("Scaling features...")
    # Scale features just for calculating fair centroids and similarities
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[feature_cols].fillna(0))
    df_scaled = pd.DataFrame(scaled_features, columns=feature_cols)
    df_scaled['track_genre'] = df['track_genre']
    
    print("Calculating global genre centroids...")
    genre_centroids = df_scaled.groupby('track_genre').mean().to_dict('index')
    
    print("Identifying duplicates...")
    # Group by name and artist (case-insensitive for safety)
    df['lower_name'] = df['track_name'].str.lower()
    df['lower_artist'] = df['artists'].str.lower()
    
    # We will keep track of which indices to KEEP
    indices_to_keep = []
    
    groups = df.groupby(['lower_name', 'lower_artist'])
    
    print(f"Processing {len(groups)} unique tracks...")
    
    # Fast iteration
    for _, group in groups:
        if len(group) == 1:
            indices_to_keep.append(group.index[0])
            continue
            
        # For duplicates, we need to find the best genre match
        # We assume audio features are identical for the same track, so we just take the first row's scaled features
        track_idx = group.index[0]
        track_features = scaled_features[track_idx]
        
        best_genre = None
        best_sim = -float('inf')
        best_row_idx = track_idx # default
        
        for idx in group.index:
            candidate_genre = df.loc[idx, 'track_genre']
            if candidate_genre in genre_centroids:
                centroid = list(genre_centroids[candidate_genre].values())
                sim = cosine_similarity([track_features], [centroid])[0][0]
                if sim > best_sim:
                    best_sim = sim
                    best_genre = candidate_genre
                    best_row_idx = idx
                    
        indices_to_keep.append(best_row_idx)

    print("Filtering dataset...")
    dedup_df = df.loc[indices_to_keep].copy()
    dedup_df = dedup_df.drop(columns=['lower_name', 'lower_artist'])
    
    new_len = len(dedup_df)
    print(f"Original tracks: {original_len}")
    print(f"Unique tracks after deduplication: {new_len}")
    print(f"Removed {original_len - new_len} duplicate entries.")
    
    # Sort for cleanliness
    dedup_df = dedup_df.sort_values(by=['track_name', 'artists'])
    
    print("Overwriting dataset...")
    dedup_df.to_csv(filepath, index=False)
    print(f"Saved deduplicated dataset to {filepath}")

if __name__ == "__main__":
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
    dataset_path = os.path.join(PROJECT_ROOT, "clustered_dataset.csv")
    
    deduplicate_dataset(dataset_path)
