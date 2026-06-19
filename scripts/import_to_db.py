import pandas as pd
import os
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Float,
    Boolean,
    ForeignKey,
)
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class Genre(Base):
    __tablename__ = "genres"
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, unique=True, nullable=False)


class Track(Base):
    __tablename__ = "tracks"
    id = Column(Integer, primary_key=True)
    track_id = Column(String, nullable=False)
    artists = Column(String)
    album_name = Column(String)
    track_name = Column(String)
    popularity = Column(Integer)
    duration_ms = Column(Integer)
    explicit = Column(Boolean)
    danceability = Column(Float)
    energy = Column(Float)
    key = Column(Integer)
    loudness = Column(Float)
    mode = Column(Integer)
    speechiness = Column(Float)
    acousticness = Column(Float)
    instrumentalness = Column(Float)
    liveness = Column(Float)
    valence = Column(Float)
    tempo = Column(Float)
    time_signature = Column(Integer)
    genre_id = Column(Integer, ForeignKey("genres.id"))


def setup_database(csv_file_path, db_file_path):
    print("--- Setting up Database ---")

    os.makedirs(os.path.dirname(db_file_path), exist_ok=True)

    engine = create_engine(f"sqlite:///{db_file_path}")
    Base.metadata.create_all(engine)

    print(f"Reading {csv_file_path}...")
    df = pd.read_csv(csv_file_path)
    df.rename(columns={df.columns[0]: "id"}, inplace=True)

    df.dropna(subset=["track_id", "track_genre"], inplace=True)

    genres_df = pd.DataFrame({"name": df["track_genre"].unique()})
    genres_df.to_sql("genres", engine, if_exists="append", index=False)

    saved_genres = pd.read_sql("genres", engine)
    genre_mapping = saved_genres.set_index("name")["id"].to_dict()

    df["genre_id"] = df["track_genre"].map(genre_mapping)
    df.drop(columns=["track_genre"], inplace=True)

    print("Importing tracks into database...")
    df.to_sql("tracks", engine, if_exists="append", index=False)
    print("Database setup complete!")


if __name__ == "__main__":
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

    csv_path = os.path.join(PROJECT_ROOT, "dataset.csv")
    db_path = os.path.join(PROJECT_ROOT, "database", "dataset.db")

    setup_database(csv_path, db_path)
