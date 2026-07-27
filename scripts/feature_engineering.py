import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity


def run_feature_engineering(
    input_filepath, output_filepath, sim_matrix_filepath, eda_out_dir
):
    print("--- Starting Feature Engineering & EDA Pipeline ---")
    print(f"Loading data from {input_filepath }...")
    df = pd.read_csv(input_filepath)
    os.makedirs(eda_out_dir, exist_ok=True)
    print("\nEngineering Interaction Terms and Anchors...")
    if "energy" in df.columns and "loudness" in df.columns:
        shifted_loudness = df["loudness"] - df["loudness"].min() + 1
        df["intensity_index"] = df["energy"] * shifted_loudness
        df["intensity_index"] = MinMaxScaler().fit_transform(df[["intensity_index"]])
        df = df.drop(columns=["loudness"])
    if "danceability" in df.columns and "tempo" in df.columns:
        df["groove_index"] = df["danceability"] * df["tempo"]
        df["groove_index"] = MinMaxScaler().fit_transform(df[["groove_index"]])
    if "valence" in df.columns and "intensity_index" in df.columns:
        df["euphoria_index"] = df["valence"] * df["intensity_index"]
        df["euphoria_index"] = MinMaxScaler().fit_transform(df[["euphoria_index"]])
    if "energy" in df.columns and "acousticness" in df.columns:
        df["electronic_index"] = df["energy"] * (1.0 - df["acousticness"])
        df["electronic_index"] = MinMaxScaler().fit_transform(df[["electronic_index"]])
    if "speechiness" in df.columns:
        df["is_spoken_word"] = (df["speechiness"] > 0.66).astype(int)
    if "instrumentalness" in df.columns:
        df["is_pure_instrumental"] = (df["instrumentalness"] > 0.50).astype(int)
    if "duration_ms" in df.columns:
        df = df.drop(columns=["duration_ms"])
    print("\nAnalyzing Genre Similarities and Generating Reports...")
    audio_features = [
        "danceability",
        "tempo",
        "valence",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "liveness",
        "energy",
        "intensity_index",
        "groove_index",
        "euphoria_index",
        "electronic_index",
        "is_spoken_word",
        "is_pure_instrumental",
    ]
    available_features = [f for f in audio_features if f in df.columns]
    if "track_genre" in df.columns and available_features:
        scaler = MinMaxScaler()
        scaled_audio = pd.DataFrame(
            scaler.fit_transform(df[available_features]), columns=available_features
        )
        scaled_audio["track_genre"] = df["track_genre"].values
        genre_centroids = scaled_audio.groupby("track_genre")[available_features].mean()
        genre_std = scaled_audio.groupby("track_genre")[available_features].std()
        import numpy as np

        semantic_weights_dict = {
            "danceability": 3.0,
            "tempo": 1.0,
            "valence": 3.0,
            "speechiness": 2.0,
            "acousticness": 2.0,
            "instrumentalness": 2.0,
            "liveness": 0.5,
            "energy": 1.0,
            "intensity_index": 1.0,
            "groove_index": 1.0,
            "euphoria_index": 1.0,
            "electronic_index": 2.0,
            "is_spoken_word": 2.0,
            "is_pure_instrumental": 2.0,
        }
        semantic_weights = np.array(
            [semantic_weights_dict.get(f, 1.0) for f in available_features]
        )
        sem_weighted_centroids = genre_centroids * semantic_weights
        print(" -> Generating Standard Similarity...")
        std_sim_matrix = pd.DataFrame(
            cosine_similarity(sem_weighted_centroids),
            index=genre_centroids.index,
            columns=genre_centroids.index,
        )
        print(" -> Generating Weighted Similarity...")
        genre_weights = (1.0 / (genre_std + 0.05)) * semantic_weights
        weighted_sim_matrix = pd.DataFrame(
            index=genre_centroids.index, columns=genre_centroids.index, dtype=float
        )
        for genre_A in genre_centroids.index:
            for genre_B in genre_centroids.index:
                if genre_A == genre_B:
                    weighted_sim_matrix.loc[genre_A, genre_B] = 1.0
                    continue
                combined_weights = (
                    genre_weights.loc[genre_A] + genre_weights.loc[genre_B]
                ) / 2.0
                vec_A = genre_centroids.loc[genre_A] * combined_weights
                vec_B = genre_centroids.loc[genre_B] * combined_weights
                sim = cosine_similarity(
                    vec_A.values.reshape(1, -1), vec_B.values.reshape(1, -1)
                )[0][0]
                weighted_sim_matrix.loc[genre_A, genre_B] = sim
        print(" -> Blending models into Ensemble Similarity...")
        ensemble_sim_matrix = (std_sim_matrix + weighted_sim_matrix) / 2.0
        with open(
            os.path.join(eda_out_dir, "engineered_ensemble_genre_similarity.txt"), "w"
        ) as f:
            f.write("--- ENGINEERED ENSEMBLE GENRE SIMILARITY (Top 3) ---\n")
            f.write(
                "50% Standard Cosine Similarity + 50% Strict-Rule Weighted Similarity\n\n"
            )
            for genre in ensemble_sim_matrix.index:
                sims = (
                    ensemble_sim_matrix.loc[genre]
                    .sort_values(ascending=False)
                    .drop(labels=[genre])
                    .head(3)
                )
                f.write(f"Genre: {genre .upper ()}\n")
                for sim_genre, score in sims.items():
                    f.write(
                        f"  -> {sim_genre .upper ()} (Ensemble Score: {score :.4f})\n"
                    )
                f.write("\n")
        top1_dict, top2_dict, top3_dict = {}, {}, {}
        for genre in ensemble_sim_matrix.index:
            top_sims = (
                ensemble_sim_matrix.loc[genre]
                .sort_values(ascending=False)
                .drop(labels=[genre])
                .head(3)
            )
            top1_dict[genre] = top_sims.index[0]
            top2_dict[genre] = top_sims.index[1]
            top3_dict[genre] = top_sims.index[2]
        df["similar_genre_1"] = df["track_genre"].map(top1_dict)
        df["similar_genre_2"] = df["track_genre"].map(top2_dict)
        df["similar_genre_3"] = df["track_genre"].map(top3_dict)
        if sim_matrix_filepath:
            os.makedirs(os.path.dirname(sim_matrix_filepath), exist_ok=True)
            ensemble_sim_matrix.to_csv(sim_matrix_filepath)
    print("\nApplying Frequency Encoding...")
    if "artists" in df.columns:
        artist_frequencies = df["artists"].value_counts().to_dict()
        df["artist_track_count"] = df["artists"].map(artist_frequencies)
    if "popularity" in df.columns:
        df["is_unpopular"] = (df["popularity"] == 0).astype(int)
    print("\nGenerating post-engineering correlation plot...")
    corr_features = available_features + [
        "popularity",
        "artist_track_count",
        "is_unpopular",
    ]
    corr_features = [f for f in corr_features if f in df.columns]
    plt.figure(figsize=(16, 12))
    sns.heatmap(
        df[corr_features].corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5
    )
    plt.title("Correlation Matrix (Including Ensemble Logic & Indexes)")
    plt.tight_layout()
    plt.savefig(os.path.join(eda_out_dir, "engineered_correlation_matrix.png"))
    plt.close()
    print(f"\nSaving engineered dataset to {output_filepath }...")
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    df.to_csv(output_filepath, index=False)
    print(f" -> Successfully saved as {output_filepath }")
    print("--- Feature Engineering Complete ---")
    return df


if __name__ == "__main__":
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
    run_feature_engineering(
        input_filepath=os.path.join(PROJECT_ROOT, "cleaned_dataset.csv"),
        output_filepath=os.path.join(PROJECT_ROOT, "engineered_dataset.csv"),
        sim_matrix_filepath=os.path.join(PROJECT_ROOT, "genre_similarity_matrix.csv"),
        eda_out_dir=os.path.join(PROJECT_ROOT, "eda_plots"),
    )
