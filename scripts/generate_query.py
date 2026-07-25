import pandas as pd
import os
import numpy as np

def generate_mixed_query():
    print("Generating a 50-song query spanning 3 distinct genres...")
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
    dataset_path = os.path.join(PROJECT_ROOT, "clustered_dataset.csv")
    
    if not os.path.exists(dataset_path):
        print(f"Error: {dataset_path} not found. Run pipeline first.")
        return
        
    df = pd.read_csv(dataset_path)
    
    # Pick tracks heavily from specific artists to test artist weighting
    artists_to_sample = {'The Beatles': 'british', 'Linkin Park': 'grunge', 'Stevie Wonder': 'funk'}
    
    selected_tracks = []
    
    for artist, genre in artists_to_sample.items():
        # Get 10 tracks by the artist
        artist_df = df[df['artists'].str.contains(artist, na=False, case=False)]
        sampled_artist = artist_df.sample(n=min(10, len(artist_df)), random_state=42)
        selected_tracks.extend(sampled_artist['track_id'].tolist())
        
        # Get 10 random tracks from the same genre that are NOT by that artist
        genre_df = df[(df['track_genre'] == genre) & (~df['artists'].str.contains(artist, na=False, case=False))]
        sampled_genre = genre_df.sample(n=min(10, len(genre_df)), random_state=42)
        selected_tracks.extend(sampled_genre['track_id'].tolist())
        
    # Also add a few random pop tracks
    pop_df = df[df['track_genre'] == 'pop']
    sampled_pop = pop_df.sample(n=5, random_state=42)
    selected_tracks.extend(sampled_pop['track_id'].tolist())
        
    # Write to query.txt
    query_path = os.path.join(PROJECT_ROOT, "query.txt")
    with open(query_path, "w") as f:
        for t in selected_tracks:
            f.write(t + "\n")
            
    print(f"Successfully wrote {len(selected_tracks)} track IDs to {query_path} (Artist Biased Query)")

if __name__ == "__main__":
    generate_mixed_query()
