# BT4103-OPTIC Environment Setup

This guide walks through setting up a reproducible Python environment for the BT4103-OPTIC project, preparing the data-processing scripts, running the Streamlit dashboard, and exploring the modelling notebooks.

## Repository Layout

- `Data_Preparation/`: raw standardisation and advanced cleaning pipelines.
- `data_preprocessing_pipeline.py`: reusable ML preprocessing pipeline that exports model-ready CSVs and metadata.
- `dashboard/`: Streamlit application for operating theatre analytics (uses Plotly for visuals).
- `Models/`: exploratory modelling scripts and notebooks (scikit-learn, CatBoost, XGBoost, etc.).

## 1. Prerequisites

- Python 3.10 (recommended to match the cached bytecode in the repo).
- `pip` 22+ (upgrade with `python -m pip install --upgrade pip`).
- Git (for cloning) and a modern shell (PowerShell, bash, or zsh).
- Optional utilities:
  - `make` (if you plan to create helper commands).
  - `conda` or `mambaforge` if you prefer Conda environments.

## 2. Clone the Project

```bash
git clone https://github.com/joshhlim/BT4103-OPTIC.git
cd BT4103-OPTIC
```

## 3. Create an Isolated Python Environment

### Option A – `venv`

```bash
python -m venv .venv

# for Windows
.venv\Scripts\activate
# for macOS/Linux
source .venv/bin/activate
```

### Option B – Conda

```bash
conda create -n optic python=3.10 -y
conda activate optic
```

> Keep the environment activated while working on the project so that `pip`, `python`, and `streamlit` commands target the right interpreter.

## 4. Install Core Dependencies

Install the packages used across the preprocessing scripts, dashboard, and modelling experiments:

```bash
pip install --upgrade pip
pip install \
    pandas \
    numpy \
    streamlit \
    plotly \
    scikit-learn \
    tqdm \
    pyarrow \
    openpyxl
```

### Optional Extras

- `catboost`, `xgboost`, `lightgbm`: required only for specific notebooks or experiments.
- `spacy` and the `en_core_web_sm` model: enable advanced text normalisation in `Data_Cleaning.py` when `USE_SPACY = True`.
- `jupyterlab` or `notebook`: for running the `.ipynb` files under `Models/`.

Install as needed:

```bash
pip install catboost xgboost lightgbm spacy jupyterlab
python -m spacy download en_core_web_sm
```

## 5. Prepare Input Data

1. Create a `Data/` folder at the repo root if it does not exist.
2. Copy the raw exports into:
   - `Data/Raw_Dataset.csv`
   - `Data/Validation_Dataset.csv`
   - Supporting legends (e.g., `Data/Legends/nature_legend.csv`), if available.
3. Adjust the `INPUT_FILE`, `OUTPUT_FILE`, and related constants at the top of:
   - `Data_Preparation/Raw_Data_Standardization.py`
   - `Data_Preparation/Data_Cleaning.py`
   - `data_preprocessing_pipeline.py`

The sample paths in the scripts point to a macOS user directory; replace them with relative paths (e.g., `"Data/Raw_Dataset.csv"`) to avoid hard-coded user names.

## 6. Run the Data Pipelines

> All commands assume your virtual environment is active and the working directory is the repository root.

1. **Standardise historical raw data** (one-time step):

   ```bash
   python Data_Preparation/Raw_Data_Standardization.py
   ```

   - Input: `Data/Raw_Dataset.csv`
   - Output: `Data/Standardized_Raw_Dataset.csv`
   - Optional: install `tqdm` to enable progress bars (Already included in the core dependencies above).

2. **Advanced cleaning and feature engineering**:

   ```bash
   python Data_Preparation/Data_Cleaning.py
   ```

   - Input: `Data/Standardized_Raw_Dataset.csv`
   - Output: `Data/Cleaned_Dataset.csv` (adjust the filename if desired).
   - Review warnings about missing reference files (e.g., legends) and ensure they exist.

3. **Model-ready preprocessing**:

   ```bash
   python data_preprocessing_pipeline.py
   ```

   - Update `INPUT_FILE` and `OUTPUT_FILE` inside the script to point to the cleaned dataset (e.g., `"Data/Cleaned_Dataset.csv"`).
   - Produces `Preprocessed_Dataset.csv` plus `Preprocessed_Dataset_metadata.json`.

4. **Model experimentation (optional)**:
   ```bash
   python Models/katelyn_mlmodel_memorysafe.py
   ```
   - Expects `Final_Cleaned_Dataset_OPTIC_7.csv` (or adjust the filename).
   - Installs of `scikit-learn`, `numpy`, `pandas` are already handled; add `catboost`/`xgboost` if running other notebooks.

## 7. Launch the Streamlit Dashboard (Localhost)

Place the dataset you want to visualise in an accessible location. Then run:

```bash
streamlit run dashboard/app.py
```

- Streamlit starts a local development server (default: `http://localhost:8501`). The terminal prints the URL; open it in your browser.
- To pin the server to a specific address/port, use for example:
  ```bash
  streamlit run dashboard/app.py --server.address=127.0.0.1 --server.port=8501
  ```
  Adjust the port if `8501` is already taken.
- Once the app loads, follow these steps:
  1. **Upload the dataset**: use the sidebar uploader to provide a CSV, compressed CSV, Parquet, or Excel file. The app reads the file into memory and validates essential columns.
  2. **Pick the reporting period**: adjust any date or time sliders to focus on a relevant window (e.g., a specific quarter).
  3. **Configure filters**: narrow down by specialties, surgeons, procedure codes, or theatre IDs to drill into the data that matters most.
  4. **Explore the tabs**:
     - The overview tab surfaces key KPIs such as utilisation rate, turnover time, and average case duration.
     - Subsequent tabs break down schedule adherence, bottleneck analysis, and predictive insights if the model outputs are present.
  5. **Download insights**: use the export/download button (where available) to capture filtered tables for reporting.
- The app caches results for faster iteration; re-upload if you need to refresh with a different dataset.
- If you encounter module import errors, double-check that the virtual environment is activated and the dependencies in Step 4 are installed.

## 8. Working with Notebooks

Install Jupyter (if you skipped earlier):

```bash
pip install jupyterlab
```

Launch it from the project root:

```bash
jupyter lab
```

Open the notebooks under `Models/` (e.g., `catboost.ipynb`, `xgboost.ipynb`). Ensure any hard-coded dataset paths inside the notebooks point to the correct files in your environment.

## 9. Common Issues and Fixes

- **FileNotFoundError**: verify the `Data/` directory and file names, or update the constants at the top of each script.
- **UnicodeDecodeError** when reading CSVs: re-export the data with UTF-8 encoding, or pass `encoding="latin-1"` to `pd.read_csv`.
- **Missing optional packages**: install them via `pip install <package>` inside the activated environment.
- **Streamlit cannot import dashboard modules**: run `streamlit` from the repository root so that `dashboard` is on the Python path.
- **Parquet reading errors**: ensure `pyarrow` is installed (already included) or install `fastparquet` as an alternative.

## 10. Next Steps

- Capture the dependency list with `pip freeze > requirements.txt` once you finalise the environment.
- Consider adding task runners (e.g., `make`, `invoke`, or `tox`) to automate the pipeline end-to-end.
- Set up Git hooks or CI workflows to lint and test scripts before deployment.

With these steps completed you should have a fully configured environment for data preparation, exploratory modelling, and dashboard analytics within the BT4103-OPTIC project.
