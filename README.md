# Earthquake Data Analysis

Clean, single-notebook Earthquake Data Analysis for the 2023 USGS catalogue.

**Group project — 3 members**

- Member  Name (Student ID)  — PLEASE REPLACE
- Member  Name (Student ID)  — PLEASE REPLACE
- Dinis Nascimento <dinisnascimento@connect.glos.ac.uk>  (detected from git history)


If you provide the full names and student IDs I will fill them in here.

## Project overview

This repository contains a self-contained analysis of the 2023 USGS earthquake
catalogue. The analysis is implemented in a single Jupyter notebook that:

- loads and cleans the raw CSV catalogue;
- engineers interpretable features (temporal, depth/magnitude categories, quality scores, regional codes);
- performs exploratory data analysis (maps, depth/magnitude analyses, quality diagnostics);
- demonstrates a simple modelling pipeline for detecting strong events (Section 9).

## Layout

- `earthquake_analysis.ipynb` — main analysis notebook with narrative, code and plots.
- `earthquake_libs.py` — shared helper module (optional imports and convenience functions).
- `data/` — input files: `earthquake_dataset.csv`, `plate_boundaries.csv`, `world_map.png`.
- `outputs/` — generated artifacts (CSV, PNG, HTML exports) created by the notebook when export flags are enabled.
- `docs/` — reserved for longer reports or exported documentation.

## Quick start

1. Create and activate a Python environment (recommended: conda):

```powershell
conda create -n earthquake python=3.11 -y
conda activate earthquake
python -m pip install -r requirements.txt
```

2. Open `earthquake_analysis.ipynb` in Jupyter Lab, Jupyter Notebook, or VS Code and run the cells.

Notes:
- The notebook is written to be tolerant of optional libraries — Plotly, Seaborn and scikit-learn are used when available but the EDA sections run without all of them.
- Section 9 (modelling) requires `scikit-learn`; export of interactive Plotly figures to PNG requires `kaleido`.

## Reproducing results and outputs

- To reproduce exported artifacts (HTML/PNG/CSV), enable the export flags in the notebook cells (Section 8.2 sets `export_epicentre_outputs`) and re-run the relevant cells. Outputs will be written to `outputs/`.
- The pipeline is deterministic given the same input CSV and environment; track `requirements.txt` and the timestamps of raw data to ensure reproducibility.

## Contributing / Git workflow

- This repository is maintained on the `main` branch. For collaborative development, create feature branches and open pull requests.
- If you want me to push changes to a branch or open a PR, tell me the branch name and message.

