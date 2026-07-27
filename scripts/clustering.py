import pandas as pd
import numpy as np
import os
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import joblib


def run_clustering(input_filepath, output_filepath, model_out_path):
    print("--- Starting Clustering Pipeline ---")
    print(f"Loading data from {input_filepath }...")
    df = pd.read_csv(input_filepath)
    vibe_features = [
        "danceability",
        "energy",
        "valence",
        "acousticness",
        "intensity_index",
        "groove_index",
        "euphoria_index",
        "electronic_index",
    ]
    available_features = [f for f in vibe_features if f in df.columns]
    print(f"Using features for clustering: {available_features }")
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[available_features].fillna(0))
    print("Evaluating optimal K for KMeans using Silhouette Score on a sample...")
    np.random.seed(42)
    sample_size = min(10000, len(scaled_features))
    idx = np.random.choice(len(scaled_features), sample_size, replace=False)
    sample_features = scaled_features[idx]
    best_k = 12
    best_score = -1
    for k in range(5, 16):
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score

        kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=5)
        labels = kmeans_temp.fit_predict(sample_features)
        score = silhouette_score(sample_features, labels)
        if score > best_score:
            best_score = score
            best_k = k
    print(f"Optimal K found: {best_k } (Silhouette Score: {best_score :.4f})")
    print(f"Fitting KMeans model with K={best_k } on full dataset...")
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    df["vibe_cluster"] = kmeans.fit_predict(scaled_features)
    unique_clusters = sorted(df["vibe_cluster"].unique())
    print(f"KMeans found {len (unique_clusters )} clusters.")
    print("Analyzing cluster centroids to generate descriptive names...")
    cluster_names = {}
    for i in unique_clusters:
        cluster_points = scaled_features[df["vibe_cluster"] == i]
        centroid = cluster_points.mean(axis=0)
        row = pd.Series(
            scaler.inverse_transform([centroid])[0], index=available_features
        )
        if row["energy"] > 0.7 and row["electronic_index"] > 0.5:
            name = "High-Energy Electronic"
        elif row["acousticness"] > 0.7:
            name = "Acoustic & Chill"
        elif row["valence"] > 0.6 and row["danceability"] > 0.6:
            name = "Upbeat & Groovy"
        elif row["energy"] > 0.6 and row["intensity_index"] > 0.6:
            name = "Intense & Loud"
        elif row["valence"] < 0.4 and row["energy"] < 0.5:
            name = "Melancholy & Low-Energy"
        else:
            max_feat = row.idxmax()
            name = f"Vibe: High {max_feat .replace ('_',' ').title ()}"
        base_name = name
        counter = 1
        while name in cluster_names.values():
            name = f"{base_name } ({counter })"
            counter += 1
        cluster_names[i] = name
        print(f"Cluster {i }: {name }")
    df["vibe_cluster_name"] = df["vibe_cluster"].map(cluster_names)
    os.makedirs(os.path.dirname(model_out_path), exist_ok=True)
    joblib.dump(
        {"kmeans": kmeans, "scaler": scaler, "cluster_names": cluster_names},
        model_out_path,
    )
    print(f"Saved clustering model to {model_out_path }")
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    df.to_csv(output_filepath, index=False)
    print(f"Saved clustered dataset to {output_filepath }")
    print("--- Clustering Pipeline Complete ---")
    return df


if __name__ == "__main__":
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
    run_clustering(
        input_filepath=os.path.join(PROJECT_ROOT, "engineered_dataset.csv"),
        output_filepath=os.path.join(PROJECT_ROOT, "clustered_dataset.csv"),
        model_out_path=os.path.join(PROJECT_ROOT, "database", "clustering_model.pkl"),
    )
