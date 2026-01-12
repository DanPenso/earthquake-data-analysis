# Earthquake Data Analysis Report (2023 USGS Catalogue)

## Table of Contents
- [Executive Summary](#executive-summary)
- [Data and Provenance](#data-and-provenance)
- [Cleaning Pipeline](#cleaning-pipeline)
- [Feature Engineering](#feature-engineering)
- [Exploratory Findings (selected)](#exploratory-findings-selected)
- [Strong-Quake Classifier](#strong-quake-classifier)
- [Limitations and Risks](#limitations-and-risks)
- [Recommended Next Steps](#recommended-next-steps)
- [Reproducibility and Deliverables](#reproducibility-and-deliverables)

## Executive Summary
- Source: USGS global earthquake catalogue for 2023 (26,642 raw rows). After cleaning, 24,432 earthquakes remain; 146 (0.6%) are strong events (magnitude >= 6.0).
- Key patterns: epicentres cluster along plate boundaries (Pacific Ring of Fire, Andean margin, Indonesian arc); shallow events dominate, but intermediate and deep foci still host strong quakes; measurement quality is mostly high with a thin low-quality tail.
- Modelling: rare-event evaluation is reported with average precision (AP) and explicit thresholds. A class-weighted logistic regression at a recall-prioritised operating point achieves precision 0.20, recall 0.83 (AP 0.46). A random forest baseline can be tuned either for recall (precision 0.29, recall 0.83; AP 0.50) or for balanced errors (best-F1: precision 0.65, recall 0.59; AP 0.50). An ablation shows that catalogue workflow fields (for example magnitude type/source and review status) substantially improve discrimination (best-F1: precision 0.58, recall 0.66; AP 0.56), indicating measurement-proxy dependence that must be interpreted cautiously.
- Reproducibility: cleaning, feature engineering, and modelling live in `Notebooks/01 Earthquake Analysis.ipynb` backed by `Notebooks/earthquakelibs.py`. Inputs sit under `Data/Raw/` (for example, `Data/Raw/Earthquake Dataset.csv`), exports under `Data/Processed/`, and environment details are logged in the notebook.

## Data and Provenance
- Dataset: USGS Earthquake Hazards Program CSV export for all 2023 events.
- Fields: timestamps, latitude/longitude, magnitude, depth, event type, and uncertainty metrics (`gap`, `rms`, `depthError`, `magError`, `horizontalError`, station counts).
- Storage: raw files in `Data/Raw/Earthquake Dataset.csv` with supporting assets (if any) in `Data/Raw/`. Outputs are written to `Data/Processed/` when export flags are enabled.

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
- Spatial patterns: epicentres align with major plate boundaries. Regional share table (from `Data/Processed/Tables/region_summary_section5_2.csv`):

| Region | Events | % Global | % Strong | Median Depth (km) |
| --- | --- | --- | --- | --- |
| Americas_west | 8,852 | 36.2 | 0.3 | 21.0 |
| Asia_WestPacific | 6,797 | 27.8 | 1.0 | 35.0 |
| Americas_east_Atlantic | 4,156 | 17.0 | 0.5 | 28.0 |
| Pacific_Oceania | 2,566 | 10.5 | 0.9 | 35.0 |
| Europe_Africa | 2,061 | 8.4 | 0.6 | 10.0 |
| unknown | 0 | 0.0 | 0.0 | 0.0 |

- Quality and uncertainty: most events have high `quality_score`, but a thin tail shows high `gap`, `magError`, or `depthError`. Low-quality tails should be down-weighted or excluded in models sensitive to measurement error.
- Visual assets: interactive globe and static PNG exported from Section 5.2 (`Data/Processed/Maps/epicentre_map_section5_2.html` / `.png`) plus the regional summary CSV above.

## Strong-Quake Classifier
- Goal: early-warning style flag for strong events (mag >= 6.0) using engineered features while handling severe class imbalance.
- Setup: stratified train/test split; preprocessing shared across models; features include physical (depth, latitude, longitude), context (`broad_region`, hemispheres, time-of-day), and quality metrics.
- Models evaluated:
  - `LogReg` (class-weighted logistic regression baseline).
  - `Forest` (compact random forest).
- Metrics (hold-out set):

| Model | Accuracy | Precision (pos) | Recall (pos) | F1 (pos) | ROC AUC | AP |
| --- | --- | --- | --- | --- | --- | --- |
| Dummy (most-frequent) | 0.994 | 0.000 | 0.000 | 0.000 | 0.500 | 0.006 |
| LogReg (recall-priority) | 0.979 | 0.198 | 0.828 | 0.320 | 0.989 | 0.463 |
| Forest B (recall-priority) | 0.987 | 0.289 | 0.828 | 0.429 | 0.973 | 0.496 |
| Forest B (best-F1) | 0.996 | 0.654 | 0.586 | 0.618 | 0.973 | 0.496 |
| Forest A (+workflow, best-F1) | 0.995 | 0.576 | 0.655 | 0.613 | 0.974 | 0.563 |

- Interpretation:
  - Average precision (AP) is preferred at 0.6% positives; ROC-AUC alone can look deceptively high.
  - Forest B illustrates the threshold trade-off: a recall-prioritised point catches more strong events but increases false alarms, while the best-F1 point improves precision at the cost of missed strong events.
  - The workflow-field ablation (Forest A vs Forest B) indicates that catalogue reporting metadata contribute materially to separability; this is treated as a catalogue-driven signal, not a causal geophysical driver.
  - Feature importances and reliability diagrams in the notebook support these interpretations and motivate geometry-aware additions (e.g., plate-boundary distance).

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
- Primary notebook: `Notebooks/01 Earthquake Analysis.ipynb` (contains cleaning, feature engineering, EDA, modelling, and export toggles).
- Shared helpers: `Notebooks/earthquakelibs.py` (imports, availability flags, project paths).
- Inputs: place the 2023 CSV in `Data/Raw/`; the notebook uses `earthquakelibs.py` paths to resolve `Data/Raw/` and `Data/Processed/`.
- Outputs (when export flags are enabled): `Data/Processed/Maps/epicentre_map_section5_2.html`, `Data/Processed/Maps/epicentre_map_section5_2.png`, `Data/Processed/Tables/region_summary_section5_2.csv`, plus any additional figures or tables generated in notebook sections.
- To publish a clean report from the notebook, render with code hidden (for example `jupyter nbconvert --to html --no-input Notebooks/01\ Earthquake\ Analysis.ipynb`) or use Quarto/nbconvert with `exclude_input=True` so the focus stays on narrative and figures.
