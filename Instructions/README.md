# Earthquake Data Analysis

Clean, single notebook earthquake data analysis for the 2023 USGS catalogue.

## Group project - 3 members

- Daniel Penson (4520042)
- Hasini Adihetty (S4530499)
- Dinis Nascimento (4540434)

## Work split and presentation flow (aligned to notebook sections)

Person 1 (Hasini Adihetty)
Theme: raw catalogue evidence, cleaning, and monitoring diagnostics  
- Raw dataset evidence → Table 1 + Table 2 (N = 26,642 + numeric summary)  
- Raw distributions + event types → Fig 1 + Fig 2 + Fig 4 (mag/depth + non-earthquake types)  
- Cleaning pipeline + audit + imbalance → Table 3 + Table 4 (24,432; strong = 146; 0.6%)  
- Temporal robustness + takeaways → Fig 20 + conclusion / limitations / next steps  
- Monitoring / station coverage diagnostics → Fig 11 + Fig 12

Person 2 (Dinis Nascimento)
Theme: feature engineering and regional/tectonic structure  
- Feature engineering overview + leakage rule → Fig 5 + Table 5  
- Epicentres + tectonics → Fig 8 (or Fig 16 if using interactive export)  
- Regional comparisons → Fig 17 + Table 7  
- Quality / uncertainty diagnostics → Fig 18  
- Problem + Objectives (not forecasting; post-event classification)

Person 3 (Daniel Penson)
Theme: depth–magnitude structure and modelling baseline  
- Depth–magnitude structure → Fig 15 + Table 6 (compact)  
- Modelling setup + split + models (LogReg vs RF vs XGB + leakage policy)  
- Model selection + baseline performance (Average Precision; what AP means under imbalance)  
- Evaluation visuals → Fig 19 (confusion, ROC, PR)

## Project overview

This repository contains a self-contained analysis of the 2023 USGS earthquake
catalogue (26,642 raw rows, cleaned to 24,432 earthquakes; 146 strong events, ~0.6%). The analysis is implemented in a single Jupyter notebook that:

- loads and cleans the raw CSV catalogue;
- engineers interpretable features (temporal, depth and magnitude categories, quality scores, regional codes);
- performs exploratory data analysis (maps, depth and magnitude analyses, quality diagnostics);
- demonstrates a modelling baseline for detecting strong events (Logistic Regression, Random Forest, XGBoost) focused on precision-recall behaviour under extreme class imbalance (best CV average precision ~0.035; hold-out precision ~0.02 and recall ~0.62 at a 0.5 threshold).

## Layout

- `Instructions/README.md`: project overview and run instructions.
- `Instructions/Requirements.txt`: minimal package list.
- `Notebooks/01 Earthquake Analysis.ipynb`: main analysis notebook with narrative, code and plots.
- `Notebooks/earthquakelibs.py`: shared helper module (optional imports and path helpers).
- `Data/Raw/`: raw inputs (Earthquake Dataset.csv, Plate Boundaries.csv).
- `Data/Processed/Images/`: report images (UniLogo.png, World Map.png).
- `Data/Processed/Maps/`: exported map HTML/PNG.
- `Data/Processed/Tables/`: exported tables and summary CSVs.

## Quick start

1. Create and activate a Python environment (recommended: conda):

```powershell
conda create -n earthquake python=3.11 -y
conda activate earthquake
python -m pip install -r Instructions/Requirements.txt
```

1. Open the notebook and run all cells:

```powershell
cd Notebooks
```

Open `01 Earthquake Analysis.ipynb` in Jupyter Lab, Jupyter Notebook, or VS Code and run all cells.

## Requirements

See `Instructions/Requirements.txt` for a minimal list of packages.

## Reproducing results and outputs

- Outputs are written to `Data/Processed/Maps` and `Data/Processed/Tables`.
- The cleaned dataset is exported to `Data/Processed/Earthquakes 2023 clean.csv`.

Note: This repo uses `Data/Processed` as the output root.

## Contributing / Git workflow

- This repository is maintained on the `main` branch. For collaborative development, create feature branches and open pull requests.
