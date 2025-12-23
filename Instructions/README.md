# Earthquake Data Analysis

Clean, single notebook earthquake data analysis for the 2023 USGS catalogue.

## Group project - 3 members

- Daniel Penson (4520042)
- Hasini Adihetty (S4530499)
- Dinis Nascimento (4540434)

## Work split (as reflected in the notebook structure)

- Hasini Adihetty: initial data exploration, cleaning narrative, and depth and magnitude EDA.
- Dinis Nascimento: feature engineering design, regional/tectonic analysis, and strong quake modelling.
- Daniel Penson: documentation updates and final QA, univariate analysis, correlation and distribution visualisations.
All reviewed the full notebook narrative and QA (hash logging, seeds, flowchart, and TOC) to ensure a consistent, submission-ready storyline.

## Project overview

This repository contains a self-contained analysis of the 2023 USGS earthquake
catalogue. The analysis is implemented in a single Jupyter notebook that:

- loads and cleans the raw CSV catalogue;
- engineers interpretable features (temporal, depth and magnitude categories, quality scores, regional codes);
- performs exploratory data analysis (maps, depth and magnitude analyses, quality diagnostics);
- demonstrates a simple modelling pipeline for detecting strong events (Section 6).

## Layout

- `Instructions/README.md`: project overview and run instructions.
- `Instructions/Requirements.txt`: minimal package list.
- `Notebooks/01 Earthquake Analysis.ipynb`: main analysis notebook with narrative, code and plots.
- `Notebooks/earthquakelibs.py`: shared helper module (optional imports and path helpers).
- `Data/Raw/`: raw inputs (Earthquake Dataset.csv, Plate Boundaries.csv).
- `Data/Processed/Images/`: report images (UniLogo.png, World Map.png).
- `Data/Processed/Figures/`: exported figures.
- `Data/Processed/Maps/`: exported map HTML/PNG.
- `Data/Processed/Tables/`: exported tables and summary CSVs.

## Quick start

1. Create and activate a Python environment (recommended: conda):

```powershell
conda create -n earthquake python=3.11 -y
conda activate earthquake
python -m pip install -r Instructions/Requirements.txt
```

2. Open the notebook and run all cells:

```powershell
cd Notebooks
```

Open `01 Earthquake Analysis.ipynb` in Jupyter Lab, Jupyter Notebook, or VS Code and run all cells.

## Requirements

See `Instructions/Requirements.txt` for a minimal list of packages.

## Reproducing results and outputs

- Outputs are written to `Data/Processed/Figures`, `Data/Processed/Maps`, and `Data/Processed/Tables`.
- The cleaned dataset is exported to `Data/Processed/Earthquakes 2023 clean.csv`.

Note: This repo uses `Data/Processed` as the output root.

## Contributing / Git workflow

- This repository is maintained on the `main` branch. For collaborative development, create feature branches and open pull requests.
