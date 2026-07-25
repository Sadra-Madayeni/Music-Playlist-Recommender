# Music Recommendation Engine Pipeline

An end-to-end automated machine learning pipeline built with Python, Pandas, Scikit-learn, and PyTorch, orchestrated by Prefect. This system processes a raw music dataset, engineers complex audio features, groups tracks by sonic similarity ("Vibes"), deduplicates noisy metadata, trains a sequence-aware LSTM model, and dynamically generates hyper-personalized, context-aware playlists.

## Pipeline Architecture

The system is fully automated and orchestrated via `run_pipeline.py`, executing the following stages sequentially:

### 1. Data Ingestion & Preprocessing
* `scripts/import_to_db.py`: Ingests the raw `dataset.csv` into a robust SQLite relational database.
* `scripts/preprocess.py`: Cleans the dataset, drops irrelevant features (e.g., time signatures, raw text modes), handles missing values, removes duplicates, and standardizes numerical scales (e.g., normalizing tempo) for machine learning compatibility.

### 2. Feature Engineering & Exploration (EDA)
* `scripts/feature_engineering.py`: Calculates complex, composite audio indices such as:
  * **Intensity Index**: Combined energy, tempo, and loudness.
  * **Groove Index**: Combined danceability, energy, and valence.
  * **Euphoria Index**: Pure upbeat sonic profiles.
* Analyzes macro-genre similarities and outputs a `genre_similarity_matrix.csv`.

### 3. Unsupervised Clustering (Vibes)
* `scripts/clustering.py`: Employs PCA and K-Means clustering across the 8 engineered audio features to map the dataset into distinct sonic "Vibes" (e.g., High-Energy Electronic, Acoustic & Chill). 

### 4. Dataset Deduplication
* `scripts/deduplicate_dataset.py`: Identifies songs listed multiple times under different genres/IDs. It calculates global genre centroids and permanently assigns the duplicate track to the single genre it mathematically fits best. This shrinks dataset bloat by over 30% and enforces strict metadata hygiene.

### 5. Sequence-Aware Model Training
* `scripts/recommender_model.py`: An advanced PyTorch LSTM neural network with custom Attention mechanisms. It parses the entire deduplicated dataset and learns contextual transition probabilities (i.e., if a user plays *Acoustic*, what do they play next?).

### 6. Playlist Generation
* `scripts/playlist_generator.py`: The final inference engine. It reads the chronological sequence in `query.txt` and dynamically triggers:
  * **The 'For You' Mix**: A global prediction of the user's next desired track based on LSTM outputs.
  * **Vibe-Specific Mixes**: Divides the query into distinct sub-tastes (e.g., 'Funk & Similar') and predicts targeted continuations within that specific cultural boundary, enforcing strict penalties against unrelated genres (e.g., aggressively penalizing K-Pop from appearing in Funk playlists).
  * **Dedicated Artist Mixes**: Tracks user loyalty. If >= 5 songs by a single artist are played, it spins off a dedicated, predictive playlist tailored exclusively to that artist's discography.

## Usage

### Step 1: Environment Setup
Install the necessary dependencies. Requires Python 3.10+:
```bash
pip install -r requirements.txt
```

### Step 2: Placing the Dataset
Ensure your raw `dataset.csv` file is placed directly in the project root directory.

### Step 3: Run the Master Pipeline
To execute the entire data science orchestrator (from database configuration to final playlist generation):
```bash
python run_pipeline.py
```

### Step 4: Generate a Custom Query (Optional)
Use the helper script to create random realistic listening sessions:
```bash
python scripts/generate_query.py
```
You can manually edit `query.txt` to test the engine's reaction to different sequences, repetitions, and genres.

### Step 5: View Results
Final personalized playlists are output directly to `generated_mixes.txt`.