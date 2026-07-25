import pandas as pd
import numpy as np
import torch
import os
import joblib
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sklearn.metrics.pairwise import cosine_similarity
from scripts.recommender_model import SessionRecommenderLSTM

def generate_mixes(query_track_ids, dataset_filepath, model_filepath, scaler_filepath):
    print("--- Starting Playlist Generation Pipeline ---")
    
    print("Loading dataset...")
    df = pd.read_csv(dataset_filepath)
    
    # Extract query tracks
    query_df = df[df['track_id'].isin(query_track_ids)]
    
    if query_df.empty:
        print("No valid tracks found for the query.")
        return
        
    print("\n--- YOUR RECENTLY PLAYED QUERY ---")
    for _, row in query_df.iterrows():
        print(f"- {row['track_name']} by {row['artists']} (Genre: {row['track_genre']})")
    print("----------------------------------\n")
    
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
    
    # Prepare query sequence
    query_features = query_df[feature_cols].fillna(0)
    scaled_query = scaler.transform(query_features)
    
    # Pad or truncate to seq_length of 5 (as trained)
    seq_length = 5
    if len(scaled_query) < seq_length:
        pad = np.zeros((seq_length - len(scaled_query), len(feature_cols)))
        scaled_query = np.vstack([pad, scaled_query])
    elif len(scaled_query) > seq_length:
        scaled_query = scaled_query[-seq_length:]
        
    seq_tensor = torch.tensor([scaled_query], dtype=torch.float32)
    
    # Predict preference vector
    with torch.no_grad():
        pref_vector = model(seq_tensor).numpy()[0]
        
    print("Predicted User Preference Vector based on session history.")
    
    # Calculate similarity across entire dataset
    print("Finding closest matches in the database...")
    all_features_scaled = scaler.transform(df[feature_cols].fillna(0))
    similarities = cosine_similarity([pref_vector], all_features_scaled)[0]
    
    # Add similarities to df
    df['similarity_score'] = similarities
    
    # Filter out songs already in query
    candidates = df[~df['track_id'].isin(query_track_ids)].copy()
    
    # --- Generate Mixes ---
    print("\n================ GENERATING MIXES ================\n")
    
    # 1. General Top Recommendations (The "For You" Mix)
    for_you = candidates.sort_values(by='similarity_score', ascending=False).head(10)
    print("* The 'For You' Mix (Closest matches to your overall vibe):")
    for _, row in for_you.iterrows():
        print(f"  - {row['track_name']} by {row['artists']} (Score: {row['similarity_score']:.2f})")
    print()
    
    # 2. Mood Mix (Based on Vibe Cluster)
    # We find the dominant cluster in the query
    if 'vibe_cluster_name' in query_df.columns:
        dominant_mood = query_df['vibe_cluster_name'].mode()[0]
        mood_mix = candidates[candidates['vibe_cluster_name'] == dominant_mood].sort_values(by='similarity_score', ascending=False).head(10)
        print(f"* The '{dominant_mood}' Mix:")
        for _, row in mood_mix.iterrows():
            print(f"  - {row['track_name']} by {row['artists']} (Score: {row['similarity_score']:.2f})")
        print()
        
    # 3. Genre Mix (Using similar_genre columns)
    # Get the most common genre in the query
    dominant_genre = query_df['track_genre'].mode()[0]
    genre_mix = candidates[
        (candidates['track_genre'] == dominant_genre) | 
        (candidates['similar_genre_1'] == dominant_genre) |
        (candidates['similar_genre_2'] == dominant_genre) |
        (candidates['similar_genre_3'] == dominant_genre)
    ].sort_values(by='similarity_score', ascending=False).head(10)
    
    print(f"* The '{dominant_genre.title()}' Mix (Including related genres):")
    for _, row in genre_mix.iterrows():
        print(f"  - {row['track_name']} by {row['artists']} (Genre: {row['track_genre']}, Score: {row['similarity_score']:.2f})")
    print()

    print("--- Playlist Generation Complete ---")

if __name__ == "__main__":
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
    
    # Mock query: Let's assume the user played some specific songs. 
    # (These IDs would normally come from a front-end)
    # I will randomly pick 5 acoustic/singer-songwriter tracks as a test
    # (Since I don't know exact IDs, I will fetch 5 acoustic IDs from the file dynamically for the test)
    
    df_temp = pd.read_csv(os.path.join(PROJECT_ROOT, "clustered_dataset.csv"))
    acoustic_tracks = df_temp[df_temp['track_genre'] == 'acoustic'].head(5)['track_id'].tolist()
    
    generate_mixes(
        query_track_ids=acoustic_tracks,
        dataset_filepath=os.path.join(PROJECT_ROOT, "clustered_dataset.csv"),
        model_filepath=os.path.join(PROJECT_ROOT, "database", "recommender_model.pth"),
        scaler_filepath=os.path.join(PROJECT_ROOT, "database", "clustering_model.pkl")
    )
