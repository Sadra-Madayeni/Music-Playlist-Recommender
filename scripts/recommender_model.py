import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import mlflow
import os
import joblib
from tqdm import tqdm


class MusicSequenceDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


class SessionRecommenderLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(SessionRecommenderLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.attention = nn.Linear(hidden_dim, 1)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        out = self.fc(context)
        return out


def create_sequences(df, feature_cols, seq_length=5):
    print(f"Loading cleaned_playlists.csv...")
    playlists = pd.read_csv("cleaned_playlists.csv")
    print("Merging playlist tracks with their scaled acoustic features...")
    feature_map = df.set_index("track_id")[feature_cols].values
    track_to_idx = {track_id: i for i, track_id in enumerate(df["track_id"].values)}
    sequences = []
    targets = []
    grouped = playlists.groupby("playlist_id")
    print(f"Creating organic human sequences of length {seq_length }...")
    for pid, group in tqdm(grouped):
        track_ids = group["track_id"].values
        valid_indices = [track_to_idx[tid] for tid in track_ids if tid in track_to_idx]
        if len(valid_indices) <= seq_length:
            continue
        playlist_features = feature_map[valid_indices]
        for i in range(len(playlist_features) - seq_length):
            seq = playlist_features[i : i + seq_length]
            target = playlist_features[i + seq_length]
            sequences.append(seq)
            targets.append(target)
            if len(sequences) >= 40000:
                break
        if len(sequences) >= 40000:
            break
    return np.array(sequences), np.array(targets)


def train_model(data_filepath, model_out_path, epochs=10, batch_size=64, lr=0.001):
    print("--- Starting Model Training Pipeline (PyTorch) ---")
    df = pd.read_csv(data_filepath)
    feature_cols = [
        "danceability",
        "energy",
        "valence",
        "acousticness",
        "intensity_index",
        "groove_index",
        "euphoria_index",
        "electronic_index",
    ]
    df[feature_cols] = df[feature_cols].fillna(0)
    scaler_path = os.path.join(
        os.path.dirname(data_filepath), "database", "clustering_model.pkl"
    )
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)["scaler"]
        scaled_data = scaler.transform(df[feature_cols])
        df_scaled = pd.DataFrame(scaled_data, columns=feature_cols)
    else:
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        df_scaled = pd.DataFrame(
            scaler.fit_transform(df[feature_cols]), columns=feature_cols
        )
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        joblib.dump({"scaler": scaler}, scaler_path)
    df_scaled["track_id"] = df["track_id"].values
    sequences, targets = create_sequences(df_scaled, feature_cols, seq_length=5)
    dataset = MusicSequenceDataset(sequences, targets)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    input_dim = len(feature_cols)
    hidden_dim = 32
    output_dim = len(feature_cols)
    model = SessionRecommenderLSTM(input_dim, hidden_dim, output_dim)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Music_Session_Recommender")
    with mlflow.start_run():
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("learning_rate", lr)
        mlflow.log_param("hidden_dim", hidden_dim)
        print(f"Training for {epochs } epochs...")
        model.train()
        for epoch in range(epochs):
            total_loss = 0
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch +1 }/{epochs }")
            for batch_seq, batch_target in progress_bar:
                optimizer.zero_grad()
                predictions = model(batch_seq)
                loss = criterion(predictions, batch_target)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                progress_bar.set_postfix(loss=loss.item())
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch +1 } Average Loss: {avg_loss :.4f}")
            mlflow.log_metric("train_loss", avg_loss, step=epoch)
        os.makedirs(os.path.dirname(model_out_path), exist_ok=True)
        torch.save(model.state_dict(), model_out_path)
        mlflow.pytorch.log_state_dict(model.state_dict(), artifact_path="model")
        print(f"Model saved to {model_out_path }")
    print("--- Model Training Complete ---")


if __name__ == "__main__":
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
    train_model(
        data_filepath=os.path.join(PROJECT_ROOT, "clustered_dataset.csv"),
        model_out_path=os.path.join(PROJECT_ROOT, "database", "recommender_model.pth"),
        epochs=5,
    )
