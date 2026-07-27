# 🎵 Phase 1 & 2 Project Walkthrough: Spotify Track Popularity & Genre Similarity AI Pipeline

An end-to-end Data Science and Machine Learning pipeline designed to ingest over 114,000 Spotify tracks, structure them into a normalized relational database, clean and normalize audio feature distributions, and engineer advanced behavioral metrics to map genre similarities using a hybrid Ensemble Model.

---

## 1. The Core Objective (The Elevator Pitch)

The primary goal of this project is twofold:

* **Unsupervised Learning (Genre Similarity & Recommendation):** Build a mathematical model that deeply understands the acoustic and behavioral DNA of music to map true genre similarities and suggest logical fallbacks (e.g., identifying the mathematically closest acoustic or electronic genre if a specific playlist style is unavailable).
* **Supervised Learning (Popularity Prediction):** Prepare a clean, normalized, and feature-engineered dataset to train Machine Learning models (such as Random Forest and XGBoost) capable of predicting a song's popularity score based on its audio profile.

---

## 2. Phase 1: Database Architecture & Automation

Instead of relying on static, unmanaged CSV files, the project implements an automated data infrastructure that reflects industry-standard data engineering practices.

* **Relational Database Normalization (SQLite & SQLAlchemy):** Raw data is ingested into a local SQLite database (`dataset.db`) and normalized into two relational tables to enforce referential integrity and eliminate string redundancy:
  * `genres`: A lookup table assigning an autoincremented integer ID to every unique genre string.
  * `tracks`: The primary dataset containing track statistics, dynamically linked to the genres table via a `genre_id` foreign key.
* **Dynamic Extraction via SQL JOINs:** The `load_data.py` script actively connects to the SQLite engine and executes an SQL `JOIN` query to reconstruct genre strings on the fly, feeding a fresh, relational DataFrame directly into the analytical pipeline.
* **The Master Orchestrator (`pipeline.py`):** A single command-line interface (`python pipeline.py`) automates the entire workflow chain: checking database initialization, executing SQL extractions, running preprocessing hygiene, generating engineered features, and exporting analytical reports.

---

## 3. Phase 2.1: Preprocessing & Data Hygiene

Raw Spotify API data contains noise, operational metadata, and duplicate entries. The `preprocess.py` module applies strict data hygiene before any mathematical modeling occurs.

* **Deduplication & Null Handling:** Automatically drops records with missing essential textual metadata (track name, artist, album) and removes duplicate rows caused by individual tracks appearing across multiple genre playlists.
* **Dimensionality Reduction (Noise Removal):** Drops operational and non-acoustic metadata columns—including `mode`, `key`, `time_signature`, `explicit`, and `duration_ms`. These variables add arbitrary numerical noise to distance algorithms (Euclidean and Cosine) without defining a song's actual musical style.
* **Scale Normalization:** Because continuous audio features operate on vastly different scales (e.g., `tempo` spans 0 to 250+ BPM, whereas `acousticness` spans 0 to 1), `MinMaxScaler` compresses all numerical features into a uniform 0 to 1 interval. This prevents large-integer columns from artificially dominating vector distance calculations.

---

## 4. Phase 2.2: Iterative Feature Engineering & Domain Breakthroughs

Rather than relying solely on raw Spotify metrics, the `feature_engineering.py` module represents an iterative, hypothesis-driven debugging process to solve classic Data Science engineering challenges.

### Iteration 1: Solving Multicollinearity
* **The Problem:** Raw `energy` and `loudness` exhibited severe collinearity ($r > 0.75$). In vector math, this gave "loudness" two identical votes, artificially pulling loud genres together regardless of their instrumentation.
* **The Fix:** Merged both attributes into a unified continuous interaction term (`intensity_index`) and discarded the raw `loudness` column.

### Iteration 2: Defeating Index Collisions (The Metal vs. EDM Bug)
* **The Problem:** Initial attempts to build behavior metrics—like `groove_index` (Danceability × Tempo)—dropped the underlying base columns. This blinded the algorithm: **Black Metal** (low danceability, extreme tempo) and **Drum and Bass** (high danceability, moderate tempo) yielded the exact same mathematical groove score (~40), causing heavy metal to cluster with electronic dance music.
* **The Fix:** Instituted a **Base + Interaction Strategy**. Keeping the raw base features alongside the engineered interactive indexes allowed the mathematics to act as a multi-dimensional weighted vector, easily separating chaotic rock from club rhythm.

### Iteration 3: Outlier Isolation via Spotify API Domain Anchors
* **The Problem:** Non-musical audio like **Comedy** (stand-up spoken word) and **Sleep** (white noise) clustered with musical genres like *Funk* and *Brazil Pop* at 94%+ confidence levels due to continuous scale sliding.
* **The Fix:** Consulted Spotify’s official API documentation and engineered strict Boolean Domain Anchors:
  * `is_spoken_word` (`speechiness > 0.66`)
  * `is_pure_instrumental` (`instrumentalness > 0.50`)
  Because Boolean flags act as massive gravitational anchors in Cosine Similarity, this instantly pushed Comedy and Podcasts to the outer fringes of the vector space, dropping their similarity scores to a realistic 60 to 70% range.

### Iteration 4: The Dubstep vs. World-Music Anomaly
* **The Problem:** Removing certain variance constraints accidentally caused **World-Music** (indigenous/folk instruments) to map directly to **Dubstep** (heavy synthetic bass drops) because both genres share low vocal metrics and unconventional pop structures.
* **The Fix:** Engineered an `electronic_index` using the formula $\text{energy} \times (1 - \text{acousticness})$. This forced high-energy synthetic laptops (Dubstep = 1.0) to violently repel high-energy organic acoustic instruments (World Music = 0.0).

### Iteration 5: The Ultimate Hybrid (Ensemble Similarity Matrix)
* **The Problem:** Model evaluations revealed a clear trade-off between algorithms:
  * **Standard Cosine Similarity (Test 1)** was the "Purist Baseline"—it beautifully grouped organic, traditional sounds (linking Blues to Soul and R&B).
  * **Inverse-Variance Weighted Similarity (Test 6/8)** was the "Strict-Rule Winner"—it cracked the code on complex electronic crossovers (permanently separating Afrobeat from Anime and organizing DJs/raves).
* **The Master Solution:** Built an **Ensemble Similarity Matrix** that calculates both algorithms independently and blends them into a 50/50 hybrid:
  $$\text{Ensemble Matrix} = \frac{\text{Standard Cosine} + \text{Weighted Cosine}}{2}$$
  This ensemble model achieved the highest overall accuracy: Blues recovered its guitar-driven acoustic roots, while EDM and rhythmic crossovers stayed rigidly locked in their correct subgenres.

---

## 5. Current Status & Next Steps

The automated pipeline currently outputs an analytical goldmine into the `eda_plots/` and root directories:
* **`engineered_dataset.csv`:** A pristine, normalized, noise-free dataset enriched with interactive indices (`groove_index`, `euphoria_index`, `electronic_index`) and domain anchors (`is_spoken_word`, `is_pure_instrumental`), perfectly primed for Machine Learning.
* **`genre_similarity_matrix.csv` & Text Reports:** Proves that the unsupervised recommendation logic successfully maps all 114 genres with high domain accuracy.