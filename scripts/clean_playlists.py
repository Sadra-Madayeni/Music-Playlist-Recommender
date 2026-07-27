import pandas as pd
import numpy as np
import os
import time

def clean_playlists():
    print("Loading valid track IDs from cleaned_dataset.csv...")
    dataset_path = "cleaned_dataset.csv"
    if not os.path.exists(dataset_path):
        print(f"Error: {dataset_path} not found.")
        return
        
    df_valid = pd.read_csv(dataset_path, usecols=['track_id'])
    valid_track_ids = set(df_valid['track_id'].dropna().unique())
    print(f"Loaded {len(valid_track_ids):,} valid track IDs.")
    
    input_file = "playlists.csv"
    output_file = "cleaned_playlists.csv"
    
    # Remove output file if exists to start fresh append
    if os.path.exists(output_file):
        os.remove(output_file)
        
    chunk_size = 10000
    total_playlists = 0
    valid_playlists = 0
    total_tracks = 0
    valid_tracks = 0
    
    print(f"Processing {input_file} in chunks of {chunk_size}...")
    start_time = time.time()
    
    # Process the CSV chunk by chunk
    # The file has no header. Column 0 is playlist ID, Columns 1-500 are track URIs
    chunk_iterator = pd.read_csv(input_file, header=None, chunksize=chunk_size, low_memory=False)
    
    first_chunk = True
    for i, chunk in enumerate(chunk_iterator):
        total_playlists += len(chunk)
        
        # 1. Melt the dataframe to long format
        # id_vars is column 0 (playlist_id)
        # value_vars are all other columns (1 to 500)
        chunk.rename(columns={0: 'playlist_id'}, inplace=True)
        melted = chunk.melt(id_vars=['playlist_id'], var_name='original_position', value_name='track_id')
        
        # 2. Drop NA values (empty track slots in the playlist)
        melted = melted.dropna(subset=['track_id'])
        total_tracks += len(melted)
        
        # 3. Strip the spotify:track: prefix and whitespaces
        melted['track_id'] = melted['track_id'].astype(str).str.replace('spotify:track:', '', regex=False).str.strip()
        
        # 4. Filter against our valid dataset
        melted = melted[melted['track_id'].isin(valid_track_ids)]
        valid_tracks += len(melted)
        
        # 5. Filter out playlists that now have fewer than 5 tracks
        counts = melted['playlist_id'].value_counts()
        valid_playlist_ids = counts[counts >= 5].index
        
        melted = melted[melted['playlist_id'].isin(valid_playlist_ids)]
        
        # Sort by playlist and original position to maintain chronological order
        melted = melted.sort_values(by=['playlist_id', 'original_position'])
        
        # Select final columns and create a clean position index
        melted['position'] = melted.groupby('playlist_id').cumcount() + 1
        final_df = melted[['playlist_id', 'track_id', 'position']]
        
        valid_playlists += len(final_df['playlist_id'].unique())
        
        # Append to CSV
        final_df.to_csv(output_file, mode='a', header=first_chunk, index=False)
        first_chunk = False
        
        if (i + 1) % 5 == 0:
            print(f"Processed {total_playlists:,} playlists... ({valid_playlists:,} kept)")
            
    end_time = time.time()
    
    print("\n" + "=" * 50)
    print("PLAYLIST CLEANING COMPLETE")
    print("=" * 50)
    print(f"Time Taken: {end_time - start_time:.2f} seconds")
    print(f"Original Playlists: {total_playlists:,}")
    if total_playlists > 0:
        print(f"Valid Playlists (>= 5 tracks): {valid_playlists:,} ({(valid_playlists/total_playlists)*100:.1f}%)")
    print(f"Original Tracks: {total_tracks:,}")
    if total_tracks > 0:
        print(f"Valid Tracks matched: {valid_tracks:,} ({(valid_tracks/total_tracks)*100:.1f}%)")
    print(f"Output saved to: {output_file}")

if __name__ == "__main__":
    clean_playlists()
