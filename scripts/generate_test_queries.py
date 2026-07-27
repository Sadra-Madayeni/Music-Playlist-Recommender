import pandas as pd
import os
import shutil
import subprocess
import sys
import random

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
dataset_path = os.path.join(PROJECT_ROOT, "clustered_dataset.csv")
print("Loading dataset...")
df = pd.read_csv(dataset_path)


def get_track_ids(condition, count=50):
    res = df[condition]
    if len(res) > count:
        res = res.sample(count, random_state=42)
    return res["track_id"].tolist()


q1 = []
q1.extend(get_track_ids(df["track_genre"] == "classical", 15))
q1.extend(get_track_ids(df["track_genre"] == "metal", 15))
q1.extend(get_track_ids(df["track_genre"] == "hip-hop", 15))
q1.extend(get_track_ids(df["track_genre"] == "acoustic", 5))
q2 = []
q2.extend(
    get_track_ids(df["artists"].str.contains("Linkin Park", na=False, case=False), 25)
)
q2.extend(
    get_track_ids(df["artists"].str.contains("The Beatles", na=False, case=False), 25)
)
q3 = get_track_ids(df["track_genre"] == "funk", 50)
q4 = get_track_ids(df["track_genre"] == "pop", 40)
rock_song = df[
    (df["track_genre"] == "rock")
    & (df["track_name"].str.contains("Hotel California", na=False, case=False))
].head(1)
if len(rock_song) == 0:
    rock_song = df[df["track_genre"] == "rock"].head(1)
rock_id = rock_song["track_id"].values[0]
q4.extend([rock_id] * 10)
lp_tracks = df[
    (df["artists"].str.contains("Linkin Park", na=False, case=False))
    & (df["track_name"].isin(["Faint", "LOST IN THE ECHO", "New Divide", "Numb"]))
]["track_id"].tolist()
q5 = []
q5.extend(lp_tracks * 5)
q5.extend(get_track_ids(df["track_genre"] == "alternative", 50 - len(q5)))
queries = {
    "1_dynamic_clustering": q1,
    "2_dedicated_artists": q2,
    "3_genre_penalty": q3,
    "4_lstm_recency": q4,
    "5_deduplication": q5,
}
rock_family = [
    "rock",
    "alt-rock",
    "hard-rock",
    "psych-rock",
    "punk",
    "grunge",
    "metal",
    "heavy-metal",
]
pop_family = ["pop", "k-pop", "mandopop", "indie-pop", "synth-pop", "dance"]
electronic_family = [
    "edm",
    "techno",
    "house",
    "trance",
    "dubstep",
    "electronic",
    "electro",
]


def sample_tracks(genres, n_total):
    tracks = []
    n_per_genre = max(1, n_total // len(genres))
    for g in genres:
        g_tracks = df[df["track_genre"] == g]["track_id"].tolist()
        if not g_tracks:
            continue
        sampled = random.sample(g_tracks, min(n_per_genre, len(g_tracks)))
        tracks.extend(sampled)
    while len(tracks) < n_total:
        g = random.choice(genres)
        g_tracks = df[df["track_genre"] == g]["track_id"].tolist()
        if g_tracks:
            tracks.append(random.choice(g_tracks))
    return tracks[:n_total]


q6 = (
    sample_tracks(rock_family, 20)
    + sample_tracks(pop_family, 15)
    + sample_tracks(electronic_family, 15)
)
random.shuffle(q6)
queries["6_realistic_diverse_genres"] = q6
query_file_path = os.path.join(PROJECT_ROOT, "query.txt")
mix_file_path = os.path.join(PROJECT_ROOT, "generated_mixes.txt")
for name, tracks in queries.items():
    print(f"\n======================================")
    print(f"Running Test Query: {name } ({len (tracks )} tracks)")
    print(f"======================================")
    with open(query_file_path, "w") as f:
        for t in tracks:
            f.write(t + "\n")
    query_backup_path = os.path.join(PROJECT_ROOT, f"query_{name }.txt")
    shutil.copy(query_file_path, query_backup_path)
    gen_script = os.path.join(CURRENT_DIR, "playlist_generator.py")
    subprocess.run([sys.executable, gen_script], check=True)
    save_path = os.path.join(PROJECT_ROOT, f"mix_{name }.txt")
    shutil.copy(mix_file_path, save_path)
    print(f"Saved results to {save_path }")
print("\nAll 5 test queries completed successfully!")
