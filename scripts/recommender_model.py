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
        # Apply custom attention or weighting to emphasize recent tracks
        self.attention = nn.Linear(hidden_dim, 1)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x shape: [batch, seq_len, features]
        lstm_out, _ = self.lstm(x) # [batch, seq_len, hidden_dim]
        
        # Calculate attention weights
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        
        # Apply weights (more recent/important tracks get higher weight)
        context = torch.sum(attn_weights * lstm_out, dim=1) # [batch, hidden_dim]
        
        out = self.fc(context) # [batch, output_dim]
        return out

def create_sequences(df, feature_cols, seq_length=5):
    # Simulate user sessions by creating rolling windows of tracks
    # Since we don't have explicit user IDs, we'll just treat contiguous blocks as sessions
    data = df[feature_cols].values
    sequences = []
    targets = []
    
    print(f"Creating sequences of length {seq_length}...")
    for i in tqdm(range(len(data) - seq_length)):
        seq = data[i:i+seq_length]
        target = data[i+seq_length]
        sequences.append(seq)
        targets.append(target)
        
        # Subsample to avoid massive dataset for local training (just take 20,000 sequences for fast local demo)
        if len(sequences) >= 20000:
            break
            
    return np.array(sequences), np.array(targets)

def train_model(data_filepath, model_out_path, epochs=10, batch_size=64, lr=0.001):
    print("--- Starting Model Training Pipeline (PyTorch) ---")
    
    # Load Data
    df = pd.read_csv(data_filepath)
    feature_cols = [
        "danceability", "energy", "valence", "acousticness", 
        "intensity_index", "groove_index", "euphoria_index", "electronic_index"
    ]
    
    # Fill missing
    df[feature_cols] = df[feature_cols].fillna(0)
    
    # We will use the same scaler from clustering
    scaler_path = os.path.join(os.path.dirname(data_filepath), "database", "clustering_model.pkl")
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)["scaler"]
        scaled_data = scaler.transform(df[feature_cols])
        df_scaled = pd.DataFrame(scaled_data, columns=feature_cols)
    else:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        df_scaled = pd.DataFrame(scaler.fit_transform(df[feature_cols]), columns=feature_cols)
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        joblib.dump({"scaler": scaler}, scaler_path) # Temp save if clustering didn't run

    sequences, targets = create_sequences(df_scaled, feature_cols, seq_length=5)
    
    dataset = MusicSequenceDataset(sequences, targets)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    input_dim = len(feature_cols)
    hidden_dim = 32
    output_dim = len(feature_cols)
    
    model = SessionRecommenderLSTM(input_dim, hidden_dim, output_dim)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Setup MLFlow tracking
    mlflow.set_experiment("Music_Session_Recommender")
    with mlflow.start_run():
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("learning_rate", lr)
        mlflow.log_param("hidden_dim", hidden_dim)
        
        print(f"Training for {epochs} epochs...")
        
        model.train()
        for epoch in range(epochs):
            total_loss = 0
            # Use tqdm for progress bar
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
            for batch_seq, batch_target in progress_bar:
                optimizer.zero_grad()
                predictions = model(batch_seq)
                loss = criterion(predictions, batch_target)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                progress_bar.set_postfix(loss=loss.item())
                
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")
            mlflow.log_metric("train_loss", avg_loss, step=epoch)
            
        # Log and save model
        os.makedirs(os.path.dirname(model_out_path), exist_ok=True)
        torch.save(model.state_dict(), model_out_path)
        # Log state dict instead of model to avoid pt2 tracing errors
        mlflow.pytorch.log_state_dict(model.state_dict(), artifact_path="model")
        
        print(f"Model saved to {model_out_path}")
        
    print("--- Model Training Complete ---")

if __name__ == "__main__":
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
    
    train_model(
        data_filepath=os.path.join(PROJECT_ROOT, "clustered_dataset.csv"),
        model_out_path=os.path.join(PROJECT_ROOT, "database", "recommender_model.pth"),
        epochs=5 # Using 5 for quick local demo
    )
