"""Shared imports and helper utilities for the Earthquake Analysis notebook.

This module centralises optional third-party imports, project paths, and small
helper functions used in the `Earthquake Analysis.ipynb` notebook. It is
designed to be safe to import even if some visualisation or ML dependencies
are missing: optional libraries are detected at import time and exposed via
the `libs` namespace along with simple availability flags.
"""
from __future__ import annotations

# Core Python stdlib imports used across the project
from pathlib import Path
import sys
import os
import json
import warnings
from datetime import datetime, timedelta

# Fundamental third-party data libraries (required)
import numpy as np
import pandas as pd

# Optional plotting and visualization libraries.
# These imports are attempted inside try/except blocks so the module
# remains importable even if plotting libraries are not available.
try:
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    from matplotlib.lines import Line2D
    HAS_MATPLOTLIB = True
except ImportError:
    plt = None
    mpimg = None
    Line2D = None
    HAS_MATPLOTLIB = False

# Seaborn provides higher-level statistical plotting convenience.
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    sns = None
    HAS_SEABORN = False

# Plotly is optional and used for interactive maps when available.
try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    px = None
    go = None
    HAS_PLOTLY = False

# Optional static export helper (Plotly image export via Kaleido).
# Kaleido is only required when exporting Plotly figures to static images.
try:
    import kaleido  # noqa: F401
    HAS_KALEIDO = True
except ImportError:
    HAS_KALEIDO = False

# Optional scientific / ML helpers (scipy, scikit-learn).
# These are attempted to provide a rich feature set for notebook
# sections that run statistical analyses or machine-learning models.
try:
    from scipy.stats import gaussian_kde
    HAS_SCIPY = True
except ImportError:
    gaussian_kde = None
    HAS_SCIPY = False

try:
    # scikit-learn offers pipeline and modelling building blocks used in
    # the notebook's modelling section. We import a broad subset so
    # that downstream cells can rely on availability checks rather
    # than importing repeatedly.
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.neighbors import BallTree
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.decomposition import PCA
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        confusion_matrix,
        classification_report,
        roc_auc_score,
        roc_curve,
        precision_recall_curve,
        average_precision_score,
        auc,
    )
    HAS_SKLEARN = True
except ImportError:
    # If sklearn is not present, expose None placeholders so callers
    # can check `libs.HAS_SKLEARN` before using ML functionality.
    train_test_split = OneHotEncoder = StandardScaler = ColumnTransformer = Pipeline = None
    SimpleImputer = None
    LogisticRegression = DecisionTreeClassifier = RandomForestClassifier = GradientBoostingClassifier = None
    BallTree = None
    KMeans = DBSCAN = PCA = None
    accuracy_score = precision_score = recall_score = f1_score = confusion_matrix = classification_report = roc_auc_score = roc_curve = None
    precision_recall_curve = average_precision_score = auc = None
    HAS_SKLEARN = False


def apply_default_plot_style() -> None:
    """Apply a default seaborn plotting style when seaborn is available.

    This is a convenience for notebook cells so they render with a
    consistent aesthetic. No-op when seaborn is not installed.
    """
    if HAS_SEABORN:
        sns.set(style="whitegrid", context="notebook")


def silence_warnings() -> None:
    """Suppress non-critical warnings to reduce notebook noise.

    Call this at runtime if you want to avoid repeated DeprecationWarning
    or UserWarning messages during exploratory analysis.
    """
    warnings.filterwarnings("ignore")


def availability() -> dict:
    """Return a dictionary summarising which optional libraries are present.

    Notebook cells use this helper to decide whether to run interactive
    visualisations or modelling blocks that depend on these packages.
    """
    return {
        "HAS_MATPLOTLIB": HAS_MATPLOTLIB,
        "HAS_SEABORN": HAS_SEABORN,
        "HAS_PLOTLY": HAS_PLOTLY,
        "HAS_KALEIDO": HAS_KALEIDO,
        "HAS_SKLEARN": HAS_SKLEARN,
        "HAS_SCIPY": HAS_SCIPY,
}


def fmt_pm(mean: float, sd: float, decimals: int = 3) -> str:
    """Format a mean +/- standard deviation string with a fixed precision."""
    return f"{mean:.{decimals}f} \u00B1 {sd:.{decimals}f}"


# Project path configuration for the reorganised repository structure.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
INSTRUCTIONS_DIR = PROJECT_ROOT / "Instructions"
NOTEBOOKS_DIR = PROJECT_ROOT / "Notebooks"
DATA_DIR = PROJECT_ROOT / "Data"
RAW_DIR = DATA_DIR / "Raw"
PROCESSED_DIR = DATA_DIR / "Processed"
FIGURES_DIR = PROCESSED_DIR / "Figures"
MAPS_DIR = PROCESSED_DIR / "Maps"
TABLES_DIR = PROCESSED_DIR / "Tables"
IMAGES_DIR = PROCESSED_DIR / "Images"

# Canonical inputs for the report.
DATA_FILE = RAW_DIR / "Earthquake Dataset.csv"
WORLD_MAP_FILE = IMAGES_DIR / "World Map.png"
UNI_LOGO_FILE = IMAGES_DIR / "UniLogo.png"
PLATE_FILE = RAW_DIR / "Plate Boundaries.csv"

def ensure_project_dirs(
    *,
    create_processed: bool = True,
    create_tables: bool = True,
    create_maps: bool = True,
    create_figures: bool = True,
    create_images: bool = True,
    raise_on_error: bool = True,
) -> dict[str, Path]:
    """Create expected output directories (explicit call; no import-time side effects).

    This function exists to keep `import earthquakelibs` safe in restricted
    environments where the repository root may not be writable.

    If directory creation fails and `raise_on_error` is False, warnings are
    emitted and the notebook can still run until it attempts an export.
    """

    targets: list[Path] = []
    if create_processed:
        targets.append(PROCESSED_DIR)
    if create_figures:
        targets.append(FIGURES_DIR)
    if create_maps:
        targets.append(MAPS_DIR)
    if create_tables:
        targets.append(TABLES_DIR)
    if create_images:
        targets.append(IMAGES_DIR)

    for path in targets:
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            msg = (
                f"Could not create directory: {path}. "
                "If the repo is not writable, set EARTHQUAKE_OUTPUT_DIR to a writable folder "
                "and rerun the notebook."
            )
            if raise_on_error:
                raise OSError(msg) from exc
            warnings.warn(msg)

    return {
        "PROCESSED_DIR": PROCESSED_DIR,
        "FIGURES_DIR": FIGURES_DIR,
        "MAPS_DIR": MAPS_DIR,
        "TABLES_DIR": TABLES_DIR,
        "IMAGES_DIR": IMAGES_DIR,
    }

# Backward-compatible alias for older notebook variables.
OUTPUTS_DIR = PROCESSED_DIR


class _Libs:
    """Lightweight namespace to expose libraries and helpers to notebooks.

    Instances of this class act as an attribute container (e.g.
    `libs.plt`, `libs.np`). The pattern simplifies optional import checks
    inside notebooks and centralises available helpers.
    """


libs = _Libs()
# Populate the `libs` namespace with references to imports, helpers and
# availability flags. Notebook cells import `earthquakelibs.libs` and
# use these attributes rather than importing packages directly.
for name, value in {
    "np": np,
    "pd": pd,
    "plt": plt,
    "sns": sns,
    "mpimg": mpimg,
    "Line2D": Line2D,
    "px": px,
    "go": go,
    "HAS_KALEIDO": HAS_KALEIDO,
    "train_test_split": train_test_split,
    "OneHotEncoder": OneHotEncoder,
    "SimpleImputer": SimpleImputer,
    "StandardScaler": StandardScaler,
    "ColumnTransformer": ColumnTransformer,
    "Pipeline": Pipeline,
    "LogisticRegression": LogisticRegression,
    "DecisionTreeClassifier": DecisionTreeClassifier,
    "RandomForestClassifier": RandomForestClassifier,
    "GradientBoostingClassifier": GradientBoostingClassifier,
    "BallTree": BallTree,
    "KMeans": KMeans,
    "DBSCAN": DBSCAN,
    "PCA": PCA,
    "accuracy_score": accuracy_score,
    "precision_score": precision_score,
    "recall_score": recall_score,
    "f1_score": f1_score,
    "confusion_matrix": confusion_matrix,
    "classification_report": classification_report,
    "roc_auc_score": roc_auc_score,
    "roc_curve": roc_curve,
    "precision_recall_curve": precision_recall_curve,
    "average_precision_score": average_precision_score,
    "auc": auc,
    "gaussian_kde": gaussian_kde,
    "warnings": warnings,
    "os": os,
    "sys": sys,
    "json": json,
    "datetime": datetime,
    "timedelta": timedelta,
    "PROJECT_ROOT": PROJECT_ROOT,
    "INSTRUCTIONS_DIR": INSTRUCTIONS_DIR,
    "NOTEBOOKS_DIR": NOTEBOOKS_DIR,
    "DATA_DIR": DATA_DIR,
    "RAW_DIR": RAW_DIR,
    "PROCESSED_DIR": PROCESSED_DIR,
    "FIGURES_DIR": FIGURES_DIR,
    "MAPS_DIR": MAPS_DIR,
    "TABLES_DIR": TABLES_DIR,
    "IMAGES_DIR": IMAGES_DIR,
    "DATA_FILE": DATA_FILE,
    "WORLD_MAP_FILE": WORLD_MAP_FILE,
    "UNI_LOGO_FILE": UNI_LOGO_FILE,
    "PLATE_FILE": PLATE_FILE,
    "OUTPUTS_DIR": OUTPUTS_DIR,
    "apply_default_plot_style": apply_default_plot_style,
    "silence_warnings": silence_warnings,
    "availability": availability,
    "fmt_pm": fmt_pm,
    "HAS_MATPLOTLIB": HAS_MATPLOTLIB,
    "HAS_SEABORN": HAS_SEABORN,
    "HAS_PLOTLY": HAS_PLOTLY,
    "HAS_SKLEARN": HAS_SKLEARN,
    "HAS_SCIPY": HAS_SCIPY,
}.items():
    setattr(libs, name, value)



# Apply the default plot style if seaborn is available. This is safe to call
# during import and will quietly continue if any backend issues occur.
try:
    apply_default_plot_style()
except Exception:
    pass


def plot_hist_with_stats(series, ax=None, title=None, xlabel=None):
    """Plot a simple histogram with mean/median reference lines."""
    if plt is None:
        return None
    ax = ax or plt.gca()
    ax.hist(series.dropna(), bins=30, color="steelblue", alpha=0.7)
    if len(series.dropna()):
        mean = series.mean()
        median = series.median()
        ax.axvline(mean, color="red", linestyle="--", label=f"Mean {mean:.2f}")
        ax.axvline(median, color="green", linestyle="-.", label=f"Median {median:.2f}")
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.legend()
    return ax


def plot_scatter_geo(df, lat_col="latitude", lon_col="longitude", color_col=None, size_col=None, **kwargs):
    """Create a Plotly scatter_geo figure if plotly is available; else return None."""
    if px is None:
        return None
    fig = px.scatter_geo(
        df,
        lat=lat_col,
        lon=lon_col,
        color=color_col,
        size=size_col,
        **kwargs,
    )
    return fig


def add_plate_distance(
    df: pd.DataFrame,
    plate_file: Path | None = PLATE_FILE,
    *,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    out_col: str = "dist_to_plate_km",
    earth_radius_km: float = 6371.0088,
) -> pd.DataFrame:
    """Add great-circle distance to nearest plate-boundary point (km).

    This is a simple, custom feature: treat the plate-boundary CSV as a cloud
    of boundary points and compute the nearest-neighbour distance from each
    earthquake epicentre using haversine geometry.

    Returns a copy with `out_col` added. If inputs/dependencies are missing,
    the column is created with NaNs so downstream code remains stable.
    """
    out = df.copy()
    out[out_col] = np.nan

    if not HAS_SKLEARN or BallTree is None:
        return out

    if plate_file is None:
        return out

    plate_path = Path(plate_file)
    if not plate_path.exists():
        return out

    plates = pd.read_csv(plate_path)
    plate_lon_col = "lon" if "lon" in plates.columns else None
    plate_lat_col = "lat" if "lat" in plates.columns else None
    if plate_lon_col is None or plate_lat_col is None:
        if plates.shape[1] >= 2:
            plate_lon_col, plate_lat_col = plates.columns[:2].tolist()
        else:
            return out

    boundary = plates[[plate_lat_col, plate_lon_col]].dropna()
    if boundary.empty:
        return out

    boundary_rad = np.deg2rad(boundary[[plate_lat_col, plate_lon_col]].to_numpy(dtype=float))
    tree = BallTree(boundary_rad, metric="haversine")

    if lat_col not in out.columns or lon_col not in out.columns:
        return out

    q = out[[lat_col, lon_col]].to_numpy(dtype=float, copy=True)
    mask = np.isfinite(q).all(axis=1)
    if not mask.any():
        return out

    q_rad = np.deg2rad(q[mask])
    dist_rad, _ = tree.query(q_rad, k=1)
    out.loc[mask, out_col] = dist_rad[:, 0] * earth_radius_km
    return out


# Expose the custom feature helper via `libs` (defined earlier in the file).
setattr(libs, "add_plate_distance", add_plate_distance)


class EarthquakePipeline:
    """Lightweight orchestrator to run the notebook steps with shared paths."""

    def __init__(
        self,
        data_file: Path = DATA_FILE,
        outputs_dir: Path = PROCESSED_DIR,
        figures_dir: Path = FIGURES_DIR,
        maps_dir: Path = MAPS_DIR,
        tables_dir: Path = TABLES_DIR,
        *,
        auto_create_dirs: bool = False,
        clean_fn=None,
        engineer_fn=None,
        train_fn=None,
        evaluate_fn=None,
    ) -> None:
        self.data_file = Path(data_file)
        self.outputs_dir = Path(outputs_dir)
        self.figures_dir = Path(figures_dir)
        self.maps_dir = Path(maps_dir)
        self.tables_dir = Path(tables_dir)
        if auto_create_dirs:
            self.ensure_dirs()
        self.clean_fn = clean_fn
        self.engineer_fn = engineer_fn
        self.train_fn = train_fn
        self.evaluate_fn = evaluate_fn
        self.raw_df = None
        self.clean_df = None
        self.feat_df = None
        self.audit_df = None

    def ensure_dirs(self) -> None:
        """Create pipeline output directories.

        Called explicitly from the notebook to avoid filesystem side effects
        during module import.
        """
        for path in (self.outputs_dir, self.figures_dir, self.maps_dir, self.tables_dir):
            path.mkdir(parents=True, exist_ok=True)

    def load(self) -> pd.DataFrame:
        """Load the raw catalogue from disk."""
        df = pd.read_csv(self.data_file, parse_dates=["time", "updated"])
        self.raw_df = df
        return df

    def clean(self, df: pd.DataFrame | None = None, **kwargs):
        """Clean the raw dataframe using the configured cleaning function."""
        if self.clean_fn is None:
            raise ValueError("clean_fn is not set on EarthquakePipeline.")
        df_in = df if df is not None else self.raw_df
        if df_in is None:
            raise ValueError("No dataframe provided to clean().")
        result = self.clean_fn(df_in, **kwargs)
        if isinstance(result, tuple):
            self.clean_df, self.audit_df = result[0], result[1]
        else:
            self.clean_df = result
        return result

    def engineer(self, df: pd.DataFrame | None = None, **kwargs) -> pd.DataFrame:
        """Engineer features using the configured feature function."""
        if self.engineer_fn is None:
            raise ValueError("engineer_fn is not set on EarthquakePipeline.")
        df_in = df if df is not None else self.clean_df
        if df_in is None:
            raise ValueError("No dataframe provided to engineer().")
        self.feat_df = self.engineer_fn(df_in, **kwargs)
        return self.feat_df

    def save_outputs(self, cleaned_df: pd.DataFrame | None = None, filename: str = "Earthquakes 2023 clean.csv") -> Path:
        """Save the cleaned dataset to the processed outputs directory."""
        df = cleaned_df if cleaned_df is not None else self.clean_df
        if df is None:
            raise ValueError("No cleaned dataframe available to save.")
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.outputs_dir / filename
        df.to_csv(out_path, index=False)
        return out_path

    def train(self, *args, **kwargs):
        """Run the training callback if configured."""
        if self.train_fn is None:
            raise ValueError("train_fn is not set on EarthquakePipeline.")
        return self.train_fn(*args, **kwargs)

    def evaluate(self, *args, **kwargs):
        """Run the evaluation callback if configured."""
        if self.evaluate_fn is None:
            raise ValueError("evaluate_fn is not set on EarthquakePipeline.")
        return self.evaluate_fn(*args, **kwargs)


__all__ = [
    "libs",
    "apply_default_plot_style",
    "silence_warnings",
    "availability",
    "fmt_pm",
    "EarthquakePipeline",
    "ensure_project_dirs",
    "PROJECT_ROOT",
    "INSTRUCTIONS_DIR",
    "NOTEBOOKS_DIR",
    "DATA_DIR",
    "RAW_DIR",
    "PROCESSED_DIR",
    "FIGURES_DIR",
    "MAPS_DIR",
    "TABLES_DIR",
    "IMAGES_DIR",
    "DATA_FILE",
    "WORLD_MAP_FILE",
    "UNI_LOGO_FILE",
    "PLATE_FILE",
    "OUTPUTS_DIR",
    "plot_hist_with_stats",
    "plot_scatter_geo",
    "add_plate_distance",
]



def main():
    """Print simple diagnostics when run as a script."""
    import platform

    print("Library availability:", availability())
    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("DATA_DIR:", DATA_DIR)
    print("RAW_DIR:", RAW_DIR)
    print("PROCESSED_DIR:", PROCESSED_DIR)
    print("FIGURES_DIR:", FIGURES_DIR)
    print("MAPS_DIR:", MAPS_DIR)
    print("TABLES_DIR:", TABLES_DIR)
    print("IMAGES_DIR:", IMAGES_DIR)
    print("Python version:", platform.python_version())


if __name__ == "__main__":
    main()
