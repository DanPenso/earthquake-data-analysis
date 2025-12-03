# Earthquake Data Analysis

Single-notebook layout: open the notebook and run the cells.

## Layout
- `earthquake_analysis.ipynb`: main analysis and write-up (includes all helper code).
- `data/earthquake_dataset.csv`, `data/plate_boundaries.csv`, `data/world_map.png`: source data and base map asset.
- `docs/`: reserved for future write-ups or reports.

## Quick start
- Install Python 3.9+ and the dependencies with `python -m pip install -r requirements.txt` (covers pandas, numpy, matplotlib, seaborn, plotly, scikit-learn, scipy, nbclient/nbconvert for execution).
- Open `earthquake_analysis.ipynb` in Jupyter or VS Code to read or re-run the analysis (map and plots render inline). Section 9 (modelling) requires scikit-learn; earlier cells run without it.
