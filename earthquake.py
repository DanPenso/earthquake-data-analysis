"""
Combined earthquake helper script
This file merges `setup_libs.py`, `debug_setup.py`, and `test_setup.py` into one script.
Run as a module or script; debug/test code runs when executed directly.
"""

# Core numerical + data manipulation
import sys
import numpy as np
import pandas as pd

# Datetime utilities
from datetime import datetime, timedelta

# -----------------------------------------------
# Visualisation libraries (wrap optional heavy imports)
# -----------------------------------------------
HAS_MATPLOTLIB = True
try:
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    from matplotlib.lines import Line2D
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
except Exception:
    plt = None
    mpimg = None
    Line2D = None
    FancyBboxPatch = None
    FancyArrowPatch = None
    HAS_MATPLOTLIB = False

# Seaborn is optional; provide a safe helper
HAS_SEABORN = True
try:
    import seaborn as sns
except Exception:
    sns = None
    HAS_SEABORN = False


def apply_default_plot_style():
    """Apply the project's default plotting style if seaborn is available."""
    if HAS_SEABORN:
        sns.set(style="whitegrid", context="notebook")
    else:
        return


# Advanced visualisation (optional)
HAS_PLOTLY = True
try:
    import plotly.express as px
    import plotly.graph_objects as go
except Exception:
    px = None
    go = None
    HAS_PLOTLY = False


# -----------------------------------------------
# Machine learning models & preprocessing (optional)
# -----------------------------------------------
HAS_SKLEARN = True
try:
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline

    # Classification models
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

    # Clustering
    from sklearn.cluster import KMeans, DBSCAN

    # Dimensionality reduction
    from sklearn.decomposition import PCA

    # Evaluation metrics
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        confusion_matrix,
        classification_report,
        roc_auc_score,
        roc_curve,
    )
except Exception:
    HAS_SKLEARN = False
    train_test_split = None
    OneHotEncoder = None
    StandardScaler = None
    ColumnTransformer = None
    Pipeline = None
    LogisticRegression = None
    DecisionTreeClassifier = None
    RandomForestClassifier = None
    GradientBoostingClassifier = None
    KMeans = None
    DBSCAN = None
    PCA = None
    accuracy_score = None
    precision_score = None
    recall_score = None
    f1_score = None
    confusion_matrix = None
    classification_report = None
    roc_auc_score = None
    roc_curve = None


# -----------------------------------------------
# Scientific computing helpers (optional)
# -----------------------------------------------
HAS_SCIPY = True
try:
    from scipy.stats import gaussian_kde
except Exception:
    gaussian_kde = None
    HAS_SCIPY = False


# -----------------------------------------------
# Utilities
# -----------------------------------------------
import warnings


def silence_warnings():
    """Call this in a notebook or script if you want to suppress warnings locally."""
    warnings.filterwarnings("ignore")


import os
import json


def availability():
    """Return a dictionary of which optional packages are available."""
    return {
        "HAS_MATPLOTLIB": HAS_MATPLOTLIB,
        "HAS_SEABORN": HAS_SEABORN,
        "HAS_PLOTLY": HAS_PLOTLY,
        "HAS_SKLEARN": HAS_SKLEARN,
        "HAS_SCIPY": HAS_SCIPY,
    }


# -----------------------------------------------
# Expose names (kept for backwards compatibility)
# -----------------------------------------------
__all__ = [
    # Core libraries
    "np", "pd",

    # Plotting / images
    "plt", "sns", "mpimg", "Line2D", "FancyBboxPatch", "FancyArrowPatch",

    # Optional advanced viz
    "px", "go",

    # Machine learning (may be None if sklearn missing)
    "train_test_split", "OneHotEncoder", "StandardScaler",
    "ColumnTransformer", "Pipeline",
    "LogisticRegression", "DecisionTreeClassifier",
    "RandomForestClassifier", "GradientBoostingClassifier",
    "KMeans", "DBSCAN", "PCA",

    # Metrics
    "accuracy_score", "precision_score", "recall_score",
    "f1_score", "confusion_matrix", "classification_report",
    "roc_auc_score", "roc_curve",

    # Scientific helpers
    "gaussian_kde",

    # Utilities & helpers
    "warnings", "os", "sys", "json", "datetime", "timedelta",
    "apply_default_plot_style", "silence_warnings", "availability",

    # Availability flags
    "HAS_MATPLOTLIB", "HAS_SEABORN", "HAS_PLOTLY", "HAS_SKLEARN", "HAS_SCIPY",
]


# -----------------------------------------------
# Debug & Test Behaviour (adapted from provided files)
# These run when the script is executed directly.
# -----------------------------------------------
def diagnose():
    """Prints diagnostic information about the environment and available packages."""
    print("Python version:", sys.version)

    # Show that availability() exists and its output
    print("\nChecking availability() from combined script:")
    try:
        avail = availability()
        print("✓ availability() returned:")
        for k, v in avail.items():
            print(f"  {k}: {v}")
    except Exception as e:
        print("✗ calling availability() raised:", e)

    # Provide some context similar to debug_setup.py
    print("\nModule context summary:")
    top_names = [
        name for name in globals().keys()
        if not name.startswith("__") and name in (
            'plt','sns','px','go','gaussian_kde','apply_default_plot_style','silence_warnings'
        )
    ]
    for n in top_names:
        print(f"  {n} = {globals().get(n)}")


def main(argv=None):
    """Simple CLI: --diagnose to show diagnostics, --apply-style to set plotting style."""
    import argparse

    parser = argparse.ArgumentParser(prog="earthquake",
                                     description="Helper module for Earthquake analysis (diagnose/apply-style)")
    parser.add_argument('--diagnose', action='store_true', help='Print environment/package availability')
    parser.add_argument('--apply-style', action='store_true', help='Apply default plot style (seaborn)')
    args = parser.parse_args(argv)

    if args.diagnose:
        diagnose()

    if args.apply_style:
        if HAS_SEABORN:
            apply_default_plot_style()
            print('Applied default seaborn plot style.')
        else:
            print('Seaborn not available; cannot apply style.')


if __name__ == "__main__":
    main()
