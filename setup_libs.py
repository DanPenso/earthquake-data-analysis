"""
Centralised library imports for the Earthquake Analytics Project.
Keeps notebooks clean, modular, and fully reproducible.
"""

# Core numerical + data manipulation
import numpy as np
import pandas as pd

# Datetime utilities (critical for feature engineering)
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
        # fallback: no-op
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
    # If scikit-learn is not installed, set names to None and mark missing
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
    """Call this in a notebook if you want to suppress warnings locally."""
    warnings.filterwarnings("ignore")


import os
import sys

# -----------------------------------------------
# Anything else used in the project
# -----------------------------------------------
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
# Expose all names when doing: from setup_libs import *
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


