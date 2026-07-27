import pandas as pd
import os
import shutil
import subprocess
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
dataset_path = os.path.join(PROJECT_ROOT, "clustered_dataset.csv")

print("Loading dataset...")
df = pd.read_csv(dataset_path)

def get_track_ids(condition, count=50):
    res = df[condition]
    if len(res) > count:
        res = res.sample(count, random_state=42)
    return res['track_id'].tolist()

# Feature 1: Dynamic Clustering (Multi-Vibe)
# 15 classical, 15 metal, 15 hip-hop, 5 acoustic
q1 = []
q1.extend(get_track_ids(df['track_genre'] == 'classical', 15))
q1.extend(get_track_ids(df['track_genre'] == 'metal', 15))
q1.extend(get_track_ids(df['track_genre'] == 'hip-hop', 15))
q1.extend(get_track_ids(df['track_genre'] == 'acoustic', 5))

# Feature 2: Dedicated Artist Mixes
# 25 Linkin Park, 25 The Beatles
q2 = []
q2.extend(get_track_ids(df['artists'].str.contains('Linkin Park', na=False, case=False), 25))
q2.extend(get_track_ids(df['artists'].str.contains('The Beatles', na=False, case=False), 25))

# Feature 3: Genre Presence Penalty
# 50 Funk songs (ensuring no k-pop in the query to test the penalty)
q3 = get_track_ids(df['track_genre'] == 'funk', 50)

# Feature 4: Temporal Sequence / LSTM Recency (Looping)
# 40 random pop songs, then 10 loops of the exact same specific rock song at the end
q4 = get_track_ids(df['track_genre'] == 'pop', 40)
rock_song = df[(df['track_genre'] == 'rock') & (df['track_name'].str.contains('Hotel California', na=False, case=False))].head(1)
if len(rock_song) == 0:
    rock_song = df[df['track_genre'] == 'rock'].head(1)
rock_id = rock_song['track_id'].values[0]
q4.extend([rock_id] * 10)

# Feature 5: Deduplication & Clean Metadata
# Focus on tracks that were previously problematic duplicates, mixed with a standard genre
lp_tracks = df[(df['artists'].str.contains('Linkin Park', na=False, case=False)) & 
               (df['track_name'].isin(['Faint', 'LOST IN THE ECHO', 'New Divide', 'Numb']))]['track_id'].tolist()
q5 = []
# Repeat the deduplicated tracks multiple times to show they don't spawn duplicates
q5.extend(lp_tracks * 5)
q5.extend(get_track_ids(df['track_genre'] == 'alternative', 50 - len(q5)))


queries = {
    "1_dynamic_clustering": q1,
    "2_dedicated_artists": q2,
    "3_genre_penalty": q3,
    "4_lstm_recency": q4,
    "5_deduplication": q5
}

query_file_path = os.path.join(PROJECT_ROOT, "query.txt")
mix_file_path = os.path.join(PROJECT_ROOT, "generated_mixes.txt")

for name, tracks in queries.items():
    print(f"\n======================================")
    print(f"Running Test Query: {name} ({len(tracks)} tracks)")
    print(f"======================================")
    
    # Write query
    with open(query_file_path, 'w') as f:
        for t in tracks:
            f.write(t + '\n')
            
    # Save a backup of the query for the user
    query_backup_path = os.path.join(PROJECT_ROOT, f"query_{name}.txt")
    shutil.copy(query_file_path, query_backup_path)
            
    # Run generator
    gen_script = os.path.join(CURRENT_DIR, "playlist_generator.py")
    subprocess.run([sys.executable, gen_script], check=True)
    
    # Move the generated output to a saved file
    save_path = os.path.join(PROJECT_ROOT, f"mix_{name}.txt")
    shutil.copy(mix_file_path, save_path)
    print(f"Saved results to {save_path}")

print("\nAll 5 test queries completed successfully!")
