import subprocess
import sys
import os


def run_script(script_path):
    """Utility function to run a script and handle errors."""
    print(f"\n{'='*50}")
    print(f"Executing: {script_path}")
    print(f"{'='*50}")

    try:
        result = subprocess.run([sys.executable, script_path], check=True)
        print(f"-> Successfully completed {script_path}\n")
    except subprocess.CalledProcessError as e:
        print(f"-> ERROR: Script {script_path} failed with exit code {e.returncode}")
        sys.exit(1)


if __name__ == "__main__":
    print("Starting Automated Data Science Pipeline...")

    scripts_dir = "scripts"

    if not os.path.exists("database/dataset.db"):
        run_script(os.path.join(scripts_dir, "import_to_db.py"))
    else:
        print("\nDatabase already exists. Skipping import step.")

    run_script(os.path.join(scripts_dir, "preprocess.py"))

    run_script(os.path.join(scripts_dir, "feature_engineering.py"))

    print("\n" + "=" * 50)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 50)
    print(
        "Check your root directory for 'engineered_dataset.csv' and the 'eda_plots/' folder."
    )
