from prefect import flow, task
import subprocess
import sys
import os


@task(name="Data Import", retries=2, retry_delay_seconds=10)
def run_import(scripts_dir):
    if not os.path.exists("database/dataset.db"):
        print("Running import_to_db.py...")
        subprocess.run(
            [sys.executable, os.path.join(scripts_dir, "import_to_db.py")], check=True
        )
    else:
        print("Database already exists. Skipping import step.")


@task(name="Data Preprocessing")
def run_preprocess(scripts_dir):
    print("Running preprocess.py...")
    subprocess.run(
        [sys.executable, os.path.join(scripts_dir, "preprocess.py")], check=True
    )


@task(name="Feature Engineering")
def run_feature_engineering(scripts_dir):
    print("Running feature_engineering.py...")
    subprocess.run(
        [sys.executable, os.path.join(scripts_dir, "feature_engineering.py")],
        check=True,
    )


@task(name="Clustering (Vibe Clusters)")
def run_clustering(scripts_dir):
    print("Running clustering.py...")
    subprocess.run(
        [sys.executable, os.path.join(scripts_dir, "clustering.py")], check=True
    )


@task(name="Deduplicate Dataset")
def run_deduplicate(scripts_dir):
    print("Running deduplicate_dataset.py...")
    subprocess.run(
        [sys.executable, os.path.join(scripts_dir, "deduplicate_dataset.py")],
        check=True,
    )


@task(name="Train Recommender Model")
def run_model_training(scripts_dir):
    print("Running recommender_model.py...")
    subprocess.run(
        [sys.executable, os.path.join(scripts_dir, "recommender_model.py")], check=True
    )


@task(name="Generate Playlists")
def run_playlist_generator(scripts_dir):
    print("Running playlist_generator.py...")
    subprocess.run(
        [sys.executable, os.path.join(scripts_dir, "playlist_generator.py")], check=True
    )


@flow(name="Music Recommendation Pipeline")
def main_pipeline():
    print("Starting Automated Data Science Pipeline with Prefect...")
    scripts_dir = "scripts"
    run_import(scripts_dir)
    run_preprocess(scripts_dir)
    run_feature_engineering(scripts_dir)
    run_clustering(scripts_dir)
    run_deduplicate(scripts_dir)
    run_model_training(scripts_dir)
    run_playlist_generator(scripts_dir)
    print("\n" + "=" * 50)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 50)


if __name__ == "__main__":
    main_pipeline()
