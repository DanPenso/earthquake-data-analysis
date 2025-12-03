"""Shared library imports and helpers for the earthquake notebook."""
from __future__ import annotations

from pathlib import Path
import sys
import os
import json
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# Optional plotting and viz libraries
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

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    sns = None
    HAS_SEABORN = False

try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    px = None
    go = None
    HAS_PLOTLY = False

# Optional static export helper (Plotly image export)
try:
    import kaleido  # noqa: F401
    HAS_KALEIDO = True
except ImportError:
    HAS_KALEIDO = False

# Optional stats/ML helpers
try:
    from scipy.stats import gaussian_kde
    HAS_SCIPY = True
except ImportError:
    gaussian_kde = None
    HAS_SCIPY = False

try:
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
    train_test_split = OneHotEncoder = StandardScaler = ColumnTransformer = Pipeline = None
    SimpleImputer = None
    LogisticRegression = DecisionTreeClassifier = RandomForestClassifier = GradientBoostingClassifier = None
    KMeans = DBSCAN = PCA = None
    accuracy_score = precision_score = recall_score = f1_score = confusion_matrix = classification_report = roc_auc_score = roc_curve = None
    precision_recall_curve = average_precision_score = auc = None
    HAS_SKLEARN = False


def apply_default_plot_style() -> None:
    if HAS_SEABORN:
        sns.set(style="whitegrid", context="notebook")


def silence_warnings() -> None:
    warnings.filterwarnings("ignore")


def availability() -> dict:
    return {
        "HAS_MATPLOTLIB": HAS_MATPLOTLIB,
        "HAS_SEABORN": HAS_SEABORN,
        "HAS_PLOTLY": HAS_PLOTLY,
        "HAS_KALEIDO": HAS_KALEIDO,
        "HAS_SKLEARN": HAS_SKLEARN,
        "HAS_SCIPY": HAS_SCIPY,
    }


class _Libs:
    """Namespace-style holder for optional libraries."""


libs = _Libs()
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


PROJECT_ROOT = Path.cwd()
if not (PROJECT_ROOT / "data").exists() and (PROJECT_ROOT.parent / "data").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent

DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

try:
    apply_default_plot_style()
except Exception:
    pass


__all__ = [
    "libs",
    "apply_default_plot_style",
    "silence_warnings",
    "availability",
    "PROJECT_ROOT",
    "DATA_DIR",
    "OUTPUTS_DIR",
]
