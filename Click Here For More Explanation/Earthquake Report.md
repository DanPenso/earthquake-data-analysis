# Earthquake Data Analysis Report (2023 USGS Catalogue)

## Executive Summary
- Source: USGS global earthquake catalogue for 2023 (26,642 raw rows). After cleaning, 24,432 earthquakes remain; 146 (0.6%) are strong events (magnitude >= 6.0).
- Key patterns: epicentres cluster along plate boundaries (Pacific Ring of Fire, Andean margin, Indonesian arc); shallow events dominate, but intermediate and deep foci still host strong quakes; measurement quality is mostly high with a thin low-quality tail.
- Modelling: a class-weighted logistic regression prioritises recall (precision 0.10, recall 0.93, PR-AUC 0.43). A compact random forest raises precision but lowers recall (precision 0.71, recall 0.17, PR-AUC 0.49). Threshold tuning and probability calibration are recommended.
- Reproducibility: cleaning, feature engineering, and modelling live in `Earthquake Analysis.ipynb` backed by `earthquakelibs.py`. Inputs sit under `data/` (for example, `data/Earthquake Dataset.csv`), exports under `outputs/`, and environment details are logged in the notebook.

## Data and Provenance
- Dataset: USGS Earthquake Hazards Program CSV export for all 2023 events.
- Fields: timestamps, latitude/longitude, magnitude, depth, event type, and uncertainty metrics (`gap`, `rms`, `depthError`, `magError`, `horizontalError`, station counts).
- Storage: raw files in `data/Earthquake Dataset.csv` with supporting assets (if any) in `data/`. Outputs are written to `outputs/` when export flags are enabled.

## Cleaning Pipeline
- Goals: retain only physically plausible earthquakes with complete core fields; document every removal.
- Steps (implemented in `clean_data()`):
  - Work on a copy; drop exact duplicate rows.
  - Keep the latest revision per `id` based on the `updated` timestamp.
  - Coerce `time` and `updated` to datetimes; numeric coercion for latitude, longitude, depth, and magnitude.
  - Drop rows missing essential fields; enforce bounds: latitude [-90, 90], longitude [-180, 180], depth [0, 700] km, magnitude [0, 10].
  - Filter to `type == "earthquake"` (remove explosions and other non-earthquake types).
- Impact on the 2023 catalogue:
  - Raw rows: 26,642; duplicates removed: 1,960.
  - Invalid coordinates, depth, or magnitude removed: 43; non-earthquake types removed: 207.
  - Final cleaned earthquakes: 24,432; strong events (>= 6.0): 146 (0.6%).

## Feature Engineering
- Implemented in `engineer_features()` to regenerate the full feature set consistently.
- Temporal: year, month, month_name, day, day_of_week/day_name, hour, part_of_day (night/morning/afternoon/evening), is_weekend, season.
- Physical severity: depth_category (shallow 0-70, intermediate 70-300, deep 300-700 km), mag_category (minor through massive), is_strong_quake (mag >= 6.0), energy_log10_J.
- Geospatial context: abs_latitude, abs_longitude, distance_from_equator_km, distance_from_prime_meridian_km, hemisphere_NS/EW, broad_region (coarse tectonic grouping).
- Data quality: boolean indicators for missing uncertainty fields, min-max normalised uncertainty metrics, composite `quality_score` (1 = best).
- Encodings: ordinal `*_code` helpers for categorical features (one-hot or embeddings recommended for modelling).

## Exploratory Findings (selected)
- Depth distribution: right-skewed; median 22 km, 10th percentile 8.7 km, 90th percentile 162.7 km, 95th percentile 319.0 km, 99th percentile 580.0 km, max 681.2 km. Negative depths are rare (~0.16%).
- Magnitude versus depth: weak linear coupling; high magnitudes occur across depth classes, so depth alone is not predictive.
- Spatial patterns: epicentres align with major plate boundaries. Regional share table (from `outputs/region_summary_section8_2.csv`):

| Region | Events | % Global | % Strong | Median Depth (km) |
| --- | --- | --- | --- | --- |
| Americas_west | 8,852 | 36.2 | 0.3 | 21.0 |
| Asia_WestPacific | 6,797 | 27.8 | 1.0 | 35.0 |
| Americas_east_Atlantic | 4,156 | 17.0 | 0.5 | 28.0 |
| Pacific_Oceania | 2,566 | 10.5 | 0.9 | 35.0 |
| Europe_Africa | 2,061 | 8.4 | 0.6 | 10.0 |
| unknown | 0 | 0.0 | 0.0 | 0.0 |

- Quality and uncertainty: most events have high `quality_score`, but a thin tail shows high `gap`, `magError`, or `depthError`. Low-quality tails should be down-weighted or excluded in models sensitive to measurement error.
- Visual assets: interactive globe and static PNG exported from Section 8.2 (`epicentre_map_section8_2.html` / `.png`) plus the regional summary CSV above.

## Strong-Quake Classifier
- Goal: early-warning style flag for strong events (mag >= 6.0) using engineered features while handling severe class imbalance.
- Setup: stratified train/test split; preprocessing shared across models; features include physical (depth, latitude, longitude), context (`broad_region`, hemispheres, time-of-day), and quality metrics.
- Models evaluated:
  - `LogReg` (class-weighted logistic regression baseline).
  - `Forest` (compact random forest).
- Metrics (hold-out set):

| Model | Accuracy | Precision (pos) | Recall (pos) | F1 (pos) | ROC AUC | PR AUC | Avg Precision | CV F1 mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LogReg | 0.948 | 0.097 | 0.931 | 0.176 | 0.986 | 0.427 | 0.436 | 0.172 (std 0.012) |
| Forest | 0.995 | 0.714 | 0.172 | 0.278 | 0.971 | 0.493 | 0.504 | n/a |

- Interpretation:
  - LogReg favours recall, suitable for high-sensitivity alerting but produces many false alarms (low precision).
  - Forest increases precision but misses more strong events (low recall). Threshold tuning is required before deployment.
  - PR-AUC is the preferred metric given the 0.6% positive rate; ROC-AUC alone overstates performance.
  - Feature importances highlight depth, latitude, and quality terms; consider adding plate-boundary distance to improve spatial signal.

## Limitations and Risks
- Reporting bias: smaller events are under-detected in sparsely instrumented regions; use magnitude-of-completeness filters for rate studies.
- Coarse region encoding: `broad_region` is longitude-driven and misses tectonic style; add plate-boundary proximity or slab depth.
- Measurement uncertainty: a low-quality tail exists; run sensitivity analyses or weight by `quality_score`.
- Imbalanced labels: strong events are rare; rely on precision/recall metrics, class weighting, and calibrated probabilities.
- Temporal artefacts: weak diurnal or weekly patterns likely reflect logging practices rather than geophysics.

## Recommended Next Steps
1) Add tectonic geometry features (distance to nearest plate boundary or slab depth) and rerun Section 8 visuals.  
2) Estimate magnitude-of-completeness by region and restrict frequency analyses to magnitudes above regional Mc.  
3) Apply temporal cross-validation and per-region evaluation for the classifier; calibrate probabilities (isotonic or Platt).  
4) Formalise quality handling: down-weight or drop the lowest-quality decile and report robustness checks.  
5) Automate monthly ingestion plus artifact generation (maps, CSVs) with a short changelog of shifts in rates or metrics.  

## Reproducibility and Deliverables
- Primary notebook: `Earthquake Analysis.ipynb` (contains cleaning, feature engineering, EDA, modelling, and export toggles).
- Shared helpers: `earthquakelibs.py` (imports, availability flags, project paths).
- Inputs: place the 2023 CSV in `data/`; the notebook auto-detects `data/` and `outputs/` via `earthquakelibs.py`.
- Outputs (when export flags are enabled): `outputs/epicentre_map_section8_2.html`, `outputs/epicentre_map_section8_2.png`, `outputs/region_summary_section8_2.csv`, plus any additional figures or tables generated in notebook sections.
- To publish a clean report from the notebook, render with code hidden (for example `jupyter nbconvert --to html --no-input Earthquake Analysis.ipynb`) or use Quarto/nbconvert with `exclude_input=True` so the focus stays on narrative and figures.