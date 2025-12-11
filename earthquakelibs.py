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
    "apply_default_plot_style": apply_default_plot_style,
    "silence_warnings": silence_warnings,
    "availability": availability,
    "HAS_MATPLOTLIB": HAS_MATPLOTLIB,
    "HAS_SEABORN": HAS_SEABORN,
    "HAS_PLOTLY": HAS_PLOTLY,
    "HAS_SKLEARN": HAS_SKLEARN,
    "HAS_SCIPY": HAS_SCIPY,
}.items():
    setattr(libs, name, value)


# Project path configuration: prefer clean, capitalised folder names but remain
# backward compatible with earlier lowercase/underscore variants.
PROJECT_ROOT = Path.cwd()
candidate_roots = [PROJECT_ROOT, PROJECT_ROOT.parent]

def _first_existing(root: Path, names: tuple[str, ...]) -> Path:
    for name in names:
        candidate = root / name
        if candidate.exists():
            return candidate
    # default to the first name under the current root if nothing exists yet
    return root / names[0]

DATA_DIR = _first_existing(PROJECT_ROOT, ("Data", "data"))
OUTPUTS_DIR = _first_existing(PROJECT_ROOT, ("Outputs", "outputs", "OutputsSourceFiles"))

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


__all__ = [
    "libs",
    "apply_default_plot_style",
    "silence_warnings",
    "availability",
    "PROJECT_ROOT",
    "DATA_DIR",
    "OUTPUTS_DIR",
    "plot_hist_with_stats",
    "plot_scatter_geo",
]


def main():
    """Print simple diagnostics when run as a script."""
    import platform

    print("Library availability:", availability())
    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("DATA_DIR:", DATA_DIR)
    print("OUTPUTS_DIR:", OUTPUTS_DIR)
    print("Python version:", platform.python_version())


if __name__ == "__main__":
    main()
