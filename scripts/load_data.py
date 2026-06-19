import pandas as pd
from database_connection import get_engine


def load_data_from_db():
    """
    Connects to the SQLite database and loads the tracks.
    Performs a SQL JOIN to replace the 'genre_id' with the actual 'track_genre' string.
    """
    print("Connecting to database...")
    engine = get_engine()

    query = """
    SELECT t.*, g.name AS track_genre
    FROM tracks t
    JOIN genres g ON t.genre_id = g.id
    """

    print("Executing SQL query...")
    df = pd.read_sql(query, engine)

    if "genre_id" in df.columns:
        df = df.drop(columns=["genre_id"])

    return df


if __name__ == "__main__":
    df = load_data_from_db()
    print(
        f"Successfully loaded {len(df)} rows and reconstructed the track_genre column."
    )
