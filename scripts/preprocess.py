import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from load_data import load_data_from_db


def run_preprocessing(output_filepath):
    print("--- Starting Data Preprocessing ---")
    print("Fetching data from SQLite database...")
    df = load_data_from_db()
    if "id" in df.columns:
        df = df.drop(columns=["id"])
        print(" -> Dropped 'id' (database index column)")
    cols_to_drop = ["mode", "key", "time_signature", "explicit", "duration_ms"]
    existing_cols_to_drop = [col for col in cols_to_drop if col in df.columns]
    if existing_cols_to_drop:
        df = df.drop(columns=existing_cols_to_drop)
        print(f" -> Dropped unnecessary ML columns: {existing_cols_to_drop }")
    initial_len = len(df)
    df = df.dropna(subset=["artists", "album_name", "track_name"])
    dropped_nas = initial_len - len(df)
    print(f" -> Dropped {dropped_nas } rows with missing essential text data")
    initial_len = len(df)
    df = df.drop_duplicates()
    dropped_dupes = initial_len - len(df)
    print(f" -> Dropped {dropped_dupes } exact duplicate rows")
    print("Normalizing numeric data scales...")
    if "tempo" in df.columns:
        scaler = MinMaxScaler()
        df[["tempo"]] = scaler.fit_transform(df[["tempo"]])
        print(" -> Normalized 'tempo' to a 0-1 scale using MinMaxScaler")
    print(f"\nSaving cleaned dataset to {output_filepath }...")
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    df.to_csv(output_filepath, index=False)
    print(f" -> Successfully saved as {output_filepath }")
    print("--- Preprocessing Complete ---")
    return df


if __name__ == "__main__":
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
    out_path = os.path.join(PROJECT_ROOT, "cleaned_dataset.csv")
    run_preprocessing(output_filepath=out_path)
