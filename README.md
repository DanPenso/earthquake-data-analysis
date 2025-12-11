# Earthquake Data Analysis

Clean, single-notebook Earthquake Data Analysis for the 2023 USGS catalogue.

**Group project - 3 members**

- Member Name (Student ID) - PLEASE REPLACE
- Hasini Adihetty (S4530499) - s4530499@glos.ac.uk
- Dinis Nascimeto (4540434) - dinisnascimento@connect.glos.ac.uk

**Work split (as reflected in the notebook structure)**
- Member Name - 
- Hasini Adihetty -
- Dinis Nascimeto - 
All three reviewed the full notebook narrative and QA (hash logging, seeds, flowchart, and TOC) to ensure a consistent, submission-ready storyline.

If you provide the full names and student IDs I will fill them in here.

## Project overview

This repository contains a self-contained analysis of the 2023 USGS earthquake
catalogue. The analysis is implemented in a single Jupyter notebook that:

- loads and cleans the raw CSV catalogue;
- engineers interpretable features (temporal, depth/magnitude categories, quality scores, regional codes);
- performs exploratory data analysis (maps, depth/magnitude analyses, quality diagnostics);
- demonstrates a simple modelling pipeline for detecting strong events (Section 9).

## Layout

- `Earthquake Analysis.ipynb` - main analysis notebook with narrative, code and plots.
- `earthquakelibs.py` - shared helper module (optional imports and convenience functions).
- `Data/` - input files: `Earthquake Dataset.csv`, `Plate Boundaries.csv`, `World Map.png`.
- `Outputs/` (or `OutputsSourceFiles/` if you prefer the legacy name) - generated artifacts (CSV, PNG, HTML exports) created by the notebook when export flags are enabled.
- `Click Here For More Explanation/Earthquake Report.md` - structured, narrative report (PhD-style) summarising data, cleaning, EDA, modelling, and outputs.

Repository hygiene: see `Ignore Rules for Project.txt` (replaces `.gitignore`). Data and outputs are excluded from version control so the repo only tracks code and docs.

## Quick start

1. Create and activate a Python environment (recommended: conda):

```powershell
conda create -n earthquake python=3.11 -y
conda activate earthquake
python -m pip install -r requirements.txt
```

2. Open `earthquake-analysis.ipynb` in Jupyter Lab, Jupyter Notebook, or VS Code and run the cells.

Notes:
- The notebook is written to be tolerant of optional libraries - Plotly, Seaborn and scikit-learn are used when available but the EDA sections run without all of them.
- Section 9 (modelling) requires `scikit-learn`; export of interactive Plotly figures to PNG requires `kaleido`.

## Requirements

Install the following packages (versions are minimums):
- numpy>=1.23
- pandas>=1.5
- matplotlib>=3.7
- seaborn>=0.12
- plotly>=5.18
- kaleido>=1.2
- scikit-learn>=1.3
- scipy>=1.10
- nbformat>=5.9
- nbclient>=0.9
- ipykernel>=6.26
- nbconvert>=7.10

## Reproducing results and outputs

- To reproduce exported artifacts (HTML/PNG/CSV), enable the export flags in the notebook cells (Section 8.2 sets `export_epicentre_outputs`) and re-run the relevant cells. Outputs will be written to `Outputs/` (or `OutputsSourceFiles/` for backward compatibility).
- The pipeline is deterministic given the same input CSV and environment; track `requirements.txt` and the timestamps of raw data to ensure reproducibility.

## Contributing / Git workflow

- This repository is maintained on the `main` branch. For collaborative development, create feature branches and open pull requests.
- If you want me to push changes to a branch or open a PR, tell me the branch name and message.
