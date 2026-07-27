import os
from sqlalchemy import create_engine


def get_engine(db_relative_path="../database/dataset.db"):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    db_full_path = os.path.join(current_dir, db_relative_path)
    engine = create_engine(f"sqlite:///{db_full_path }")
    return engine
