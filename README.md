## Step-by-Step Installation & Execution

### Step 1: Environment Setup
Ensure you are running Python 3.10 or higher. Install the mandatory packages from the root directory:
```bash
pip install -r requirements.txt
```

### Step 2: Placing the Dataset
Ensure your raw `dataset.csv` file is placed directly in the project root directory.

### Step 3: Run the Master Pipeline
To execute the database configuration, preprocessing, and feature engineering steps sequentially, run the master orchestrator script:
```bash
python pipeline.py
```