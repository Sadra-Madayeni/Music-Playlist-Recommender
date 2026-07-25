import pandas as pd
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib

def run_clustering(input_filepath, output_filepath, model_out_path):
    print("--- Starting Clustering Pipeline ---")
    
    print(f"Loading data from {input_filepath}...")
    df = pd.read_csv(input_filepath)
    
    # Select features that define a song's "vibe" or mood
    vibe_features = [
        "danceability", "energy", "valence", "acousticness", 
        "intensity_index", "groove_index", "euphoria_index", "electronic_index"
    ]
    
    available_features = [f for f in vibe_features if f in df.columns]
    
    print(f"Using features for clustering: {available_features}")
    
    # Scale features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[available_features].fillna(0))
    
    # Run K-Means
    n_clusters = 12 # Let's define 12 distinct vibe clusters
    print(f"Running K-Means clustering with k={n_clusters}...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['vibe_cluster'] = kmeans.fit_predict(scaled_features)
    
    # Analyze the clusters to give them descriptive names
    print("Analyzing cluster centroids to generate descriptive names...")
    centroids = scaler.inverse_transform(kmeans.cluster_centers_)
    cluster_df = pd.DataFrame(centroids, columns=available_features)
    
    cluster_names = {}
    for i in range(n_clusters):
        row = cluster_df.iloc[i]
        # Very simple naming logic based on dominant features
        if row['energy'] > 0.7 and row['electronic_index'] > 0.5:
            name = "High-Energy Electronic"
        elif row['acousticness'] > 0.7:
            name = "Acoustic & Chill"
        elif row['valence'] > 0.6 and row['danceability'] > 0.6:
            name = "Upbeat & Groovy"
        elif row['energy'] > 0.6 and row['intensity_index'] > 0.6:
            name = "Intense & Loud"
        elif row['valence'] < 0.4 and row['energy'] < 0.5:
            name = "Melancholy & Low-Energy"
        else:
            # Fallback based on max feature
            max_feat = row.idxmax()
            name = f"Vibe: High {max_feat.replace('_', ' ').title()}"
        
        # Ensure uniqueness if there are collisions
        base_name = name
        counter = 1
        while name in cluster_names.values():
            name = f"{base_name} ({counter})"
            counter += 1
            
        cluster_names[i] = name
        print(f"Cluster {i}: {name}")
        
    df['vibe_cluster_name'] = df['vibe_cluster'].map(cluster_names)
    
    # Save the model and scaler
    os.makedirs(os.path.dirname(model_out_path), exist_ok=True)
    joblib.dump({"kmeans": kmeans, "scaler": scaler, "cluster_names": cluster_names}, model_out_path)
    print(f"Saved clustering model to {model_out_path}")
    
    # Save the clustered dataset
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    df.to_csv(output_filepath, index=False)
    print(f"Saved clustered dataset to {output_filepath}")
    
    print("--- Clustering Pipeline Complete ---")
    return df

if __name__ == "__main__":
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
    
    run_clustering(
        input_filepath=os.path.join(PROJECT_ROOT, "engineered_dataset.csv"),
        output_filepath=os.path.join(PROJECT_ROOT, "clustered_dataset.csv"),
        model_out_path=os.path.join(PROJECT_ROOT, "database", "clustering_model.pkl")
    )
