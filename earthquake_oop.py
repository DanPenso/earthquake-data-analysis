"""Shared library imports and helpers for the earthquake notebook with OOP structure.

This module centralises optional third-party imports, helper functions, and
OOP-based classes for earthquake data processing and visualization. The file is
designed to be safe to import even when some visualization or ML dependencies
are missing: optional libraries are detected at import time and exposed via the
`libs` namespace along with availability flags.

The module provides two main classes:
- EarthquakeDataset: Encapsulates data cleaning and feature engineering
- EarthquakeVisualizer: Provides reusable plotting methods for earthquake data

Do not change runtime behaviour; this module provides clean, modular code
with inline comments explaining each logical block for maintainability.
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
from pandas.api.types import is_datetime64_any_dtype

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
# availability flags. Notebook cells import `earthquake_oop.libs` and
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


# Project path configuration: prefer a workspace-local `data/` folder.
PROJECT_ROOT = Path.cwd()
if not (PROJECT_ROOT / "data").exists() and (PROJECT_ROOT.parent / "data").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent

DATA_DIR = PROJECT_ROOT / "data"  # location of CSVs and static assets
OUTPUTS_DIR = PROJECT_ROOT / "outputs"  # where exported figures and tables are written

# Apply the default plot style if seaborn is available. This is safe to call
# during import and will quietly continue if any backend issues occur.
try:
    apply_default_plot_style()
except Exception:
    pass


# ============================================================================
# OOP CLASSES FOR EARTHQUAKE DATA ANALYSIS
# ============================================================================


class EarthquakeDataset:
    """Object-oriented interface for earthquake data cleaning and feature engineering.

    This class encapsulates the logic for preparing raw earthquake data by:
    - Cleaning: removing duplicates, validating ranges, handling missing values
    - Engineering: creating temporal, physical, spatial, and quality-based features

    Attributes:
        raw_df (pd.DataFrame): The original input dataframe (read-only access via property)
        _cleaned_df (pd.DataFrame): Internal storage of cleaned dataframe (private)
        _featured_df (pd.DataFrame): Internal storage of engineered dataframe (private)
    """

    def __init__(self, df: pd.DataFrame):
        """Initialize the dataset with a raw earthquake dataframe.

        Args:
            df (pd.DataFrame): Raw earthquake data from USGS or similar source.
        """
        self._raw_df = df.copy()  # Private: store original raw data
        self._cleaned_df = None  # Private: will store cleaned data after clean() is called
        self._featured_df = None  # Private: will store engineered features after engineer_features()

    @property
    def raw_df(self) -> pd.DataFrame:
        """Public read-only access to the original raw dataframe."""
        return self._raw_df

    @property
    def cleaned_df(self) -> pd.DataFrame:
        """Public read-only access to the cleaned dataframe.

        Returns None if clean() has not been called yet.
        """
        return self._cleaned_df

    @property
    def featured_df(self) -> pd.DataFrame:
        """Public read-only access to the engineered features dataframe.

        Returns None if engineer_features() has not been called yet.
        """
        return self._featured_df

    def clean(self) -> pd.DataFrame:
        """Public method: Clean the raw earthquake dataframe.

        Removes duplicates, validates ranges, converts types, and filters to earthquakes only.
        Stores result in _cleaned_df and returns it.

        Returns:
            pd.DataFrame: The cleaned earthquake dataframe.
        """
        df = self._raw_df.copy()
        original_rows = len(df)

        # Remove exact duplicates
        df = df.drop_duplicates()
        print(f"Removed {original_rows - len(df)} exact duplicate rows.")

        # Convert time fields to datetime
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df["updated"] = pd.to_datetime(df["updated"], errors="coerce")

        # Remove duplicate earthquake IDs, keeping most recent
        rows_before_id_clean = len(df)
        df = (
            df.sort_values("updated")
            .drop_duplicates(subset="id", keep="last")
        )
        print(f"Removed {rows_before_id_clean - len(df)} rows with duplicate earthquake based on IDs.")

        # Ensure numeric columns are numeric
        for col in ["latitude", "longitude", "depth", "mag"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Remove rows with missing essential fields
        essential_cols = ["time", "latitude", "longitude", "depth", "mag"]
        rows_before_essential_clean = len(df)
        df = df.dropna(subset=essential_cols)
        print(f"Removed {rows_before_essential_clean - len(df)} rows with missing essential earthquake information in {essential_cols}.")

        # Apply geophysical validity checks
        rows_before_ranges = len(df)
        valid_lat = df["latitude"].between(-90, 90)
        valid_lon = df["longitude"].between(-180, 180)
        valid_depth = df["depth"].between(0, 700)
        valid_mag = df["mag"].between(0, 10)
        df = df[valid_lat & valid_lon & valid_depth & valid_mag]
        print(f"Removed {rows_before_ranges - len(df)} rows with invalid geographical ranges.")

        # Filter to earthquakes only
        if 'type' in df.columns:
            rows_before_type_clean = len(df)
            df = df[df["type"] == "earthquake"]
            print(f"Removed {rows_before_type_clean - len(df)} non earthquake events.")

        self._cleaned_df = df  # Private: store cleaned data internally
        return self._cleaned_df

    def engineer_features(self) -> pd.DataFrame:
        """Public method: Engineer temporal, physical, spatial, and quality features.

        Requires that clean() has been called first. Creates new derived features
        from the cleaned dataframe and stores result in _featured_df.

        Returns:
            pd.DataFrame: The dataframe enriched with engineered features.

        Raises:
            ValueError: If clean() has not been called yet.
        """
        if self._cleaned_df is None:
            raise ValueError("Must call clean() before engineer_features(). Run: dataset.clean()")

        df = self._cleaned_df.copy()

        # ===== 1. TEMPORAL FEATURES =====
        df = self._engineer_temporal_features(df)

        # ===== 2. PHYSICAL SEVERITY =====
        df = self._engineer_physical_features(df)

        # ===== 3. GEOGRAPHICAL CONTEXT =====
        df = self._engineer_geographical_features(df)

        # ===== 4. DATA QUALITY =====
        df = self._engineer_quality_features(df)

        # ===== 5. CATEGORICAL ENCODINGS =====
        df = self._encode_categories(df)

        self._featured_df = df  # Private: store engineered data internally
        return self._featured_df

    # ========== PRIVATE HELPER METHODS FOR FEATURE ENGINEERING ==========

    def _engineer_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Protected method: Engineer temporal features (private helper).

        Creates year, month, day, hour, season, and related temporal indicators.
        """
        if not is_datetime64_any_dtype(df["time"]):
            df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True)

        dt = df["time"].dt
        df["year"] = dt.year
        df["month"] = dt.month
        df["month_name"] = dt.month_name()
        df["day"] = dt.day
        df["day_of_week"] = dt.dayofweek
        df["day_name"] = dt.day_name()
        df["hour"] = dt.hour
        df["is_weekend"] = df["day_of_week"].isin([5, 6])

        df["part_of_day"] = df["hour"].apply(self._hour_to_part)
        df["season"] = df["month"].apply(self._month_to_season)

        return df

    @staticmethod
    def _hour_to_part(h: int) -> str:
        """Protected static method: Map hour to part of day (private helper)."""
        if h < 6:
            return "night"
        if h < 12:
            return "morning"
        if h < 18:
            return "afternoon"
        return "evening"

    @staticmethod
    def _month_to_season(m: int) -> str:
        """Protected static method: Map month to season (private helper)."""
        if m in (12, 1, 2):
            return "winter"
        if m in (3, 4, 5):
            return "spring"
        if m in (6, 7, 8):
            return "summer"
        return "autumn"

    def _engineer_physical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Protected method: Engineer physical severity features (private helper)."""
        df["depth_category"] = pd.cut(
            df["depth"],
            bins=[0, 70, 300, 700],
            labels=["shallow", "intermediate", "deep"],
            right=False
        )
        df["mag_category"] = pd.cut(
            df["mag"],
            bins=[0, 3, 4, 5, 6, 7, 8, 10],
            labels=["minor", "light", "moderate", "strong", "major", "great", "massive"],
            right=False
        )
        df["is_strong_quake"] = df["mag"] >= 6.0
        df["energy_log10_J"] = 1.5 * df["mag"] + 4.8

        return df

    def _engineer_geographical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Protected method: Engineer geographical context features (private helper)."""
        df["abs_latitude"] = df["latitude"].abs()
        df["abs_longitude"] = df["longitude"].abs()
        df["distance_from_equator_km"] = df["abs_latitude"] * 111.0
        df["distance_from_prime_meridian_km"] = (
            df["abs_longitude"] * 111.0 * np.cos(np.deg2rad(df["latitude"]))
        )
        df["hemisphere_NS"] = np.where(df["latitude"] >= 0, "north", "south")
        df["hemisphere_EW"] = np.where(df["longitude"] >= 0, "east", "west")
        df["broad_region"] = df["longitude"].apply(self._classify_region)

        return df

    @staticmethod
    def _classify_region(lon: float) -> str:
        """Protected static method: Classify longitude into broad region (private helper)."""
        if lon < -100:
            return "Americas_west"
        if lon < -30:
            return "Americas_east_Atlantic"
        if lon < 60:
            return "Europe_Africa"
        if lon < 150:
            return "Asia_WestPacific"
        return "Pacific_Oceania"

    def _engineer_quality_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Protected method: Engineer data quality features (private helper)."""
        for col in ["depthError", "magError", "horizontalError"]:
            if col in df.columns:
                df[f"has_{col}"] = df[col].notna()

        norm_cols = []
        for col in ["gap", "rms", "depthError", "magError", "horizontalError"]:
            if col in df.columns:
                mn, mx = df[col].min(), df[col].max()
                if pd.notna(mn) and pd.notna(mx) and mn != mx:
                    norm_name = f"{col}_norm"
                    df[norm_name] = (df[col] - mn) / (mx - mn)
                    norm_cols.append(norm_name)

        if norm_cols:
            df["quality_score"] = 1 - df[norm_cols].mean(axis=1)

        return df

    def _encode_categories(self, df: pd.DataFrame) -> pd.DataFrame:
        """Protected method: Encode categorical features as integers (private helper)."""
        cat_cols = [
            "depth_category", "mag_category", "hemisphere_NS",
            "hemisphere_EW", "broad_region", "part_of_day", "season"
        ]
        for c in cat_cols:
            if c in df.columns:
                df[f"{c}_code"] = df[c].astype("category").cat.codes

        return df


class EarthquakeVisualizer:
    """Object-oriented interface for earthquake data visualization.

    This class encapsulates plotting methods for earthquake data. Each method
    creates a specific type of plot without modifying the underlying data.

    Attributes:
        _df (pd.DataFrame): Internal storage of dataframe to visualize (private)
    """

    def __init__(self, df: pd.DataFrame):
        """Initialize the visualizer with an earthquake dataframe.

        Args:
            df (pd.DataFrame): Earthquake dataframe to visualize (cleaned or featured).
        """
        self._df = df.copy()  # Private: store dataframe for plotting

    @property
    def df(self) -> pd.DataFrame:
        """Public read-only access to the dataframe being visualized."""
        return self._df

    def depth_histogram(self, bins: int = 20, figsize: tuple = (8, 5)) -> None:
        """Public method: Plot histogram of earthquake depths.

        Args:
            bins (int): Number of histogram bins. Default 20.
            figsize (tuple): Figure size as (width, height). Default (8, 5).
        """
        if not HAS_MATPLOTLIB:
            print("Matplotlib not available for plotting.")
            return

        plt.figure(figsize=figsize)
        plt.hist(self._df['depth'], bins=bins, color='red', edgecolor='black', alpha=0.7)
        plt.xlim(0, 700)
        plt.title('Distribution of Earthquake Depths', fontsize=14)
        plt.xlabel('Depth (km)')
        plt.ylabel('Frequency')
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.show()

    def epicentre_scatter(self, figsize: tuple = (10, 6), save: bool = False) -> None:
        """Public method: Plot global distribution of earthquake epicentres.

        Args:
            figsize (tuple): Figure size as (width, height). Default (10, 6).
            save (bool): Whether to save the plot to OUTPUTS_DIR. Default False.
        """
        if not HAS_MATPLOTLIB:
            print("Matplotlib not available for plotting.")
            return

        plt.figure(figsize=figsize)
        plt.scatter(self._df['longitude'], self._df['latitude'], s=1, alpha=0.5, c='darkblue')
        plt.xlim(-180, 180)
        plt.ylim(-90, 90)
        plt.title('Global Distribution of Earthquake Epicenters', fontsize=14)
        plt.xlabel('Longitude (Degrees)')
        plt.ylabel('Latitude (Degrees)')
        plt.grid(True, linestyle='--', alpha=0.7)

        if save and OUTPUTS_DIR is not None:
            OUTPUTS_DIR.mkdir(exist_ok=True)
            plt.savefig(OUTPUTS_DIR / "epicentre_scatter.png", dpi=150)
            print(f"Saved epicentre scatter to {OUTPUTS_DIR / 'epicentre_scatter.png'}")

        plt.tight_layout()
        plt.show()

    def type_barplot(self, figsize: tuple = (8, 6)) -> None:
        """Public method: Plot distribution of earthquake event types.

        Args:
            figsize (tuple): Figure size as (width, height). Default (8, 6).
        """
        if not HAS_MATPLOTLIB:
            print("Matplotlib not available for plotting.")
            return

        if 'type' not in self._df.columns:
            print("Column 'type' not found in dataframe.")
            return

        type_counts = self._df['type'].value_counts()
        plt.figure(figsize=figsize)
        ax = type_counts.plot(kind='bar', color='steelblue', edgecolor='black', alpha=0.7)
        plt.title('Distribution of Event Types', fontsize=14)
        plt.xlabel('Event Type')
        plt.ylabel('Frequency')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.show()

    def station_count_comparison(self, figsize: tuple = (10, 5)) -> None:
        """Public method: Plot distribution of number of stations reporting events.

        Args:
            figsize (tuple): Figure size as (width, height). Default (10, 5).
        """
        if not HAS_MATPLOTLIB:
            print("Matplotlib not available for plotting.")
            return

        if 'nst' not in self._df.columns:
            print("Column 'nst' (station count) not found in dataframe.")
            return

        nst_data = self._df['nst'].dropna()
        plt.figure(figsize=figsize)
        plt.hist(nst_data, bins=30, color='green', edgecolor='black', alpha=0.7)
        plt.title('Distribution of Number of Stations Reporting Events', fontsize=14)
        plt.xlabel('Number of Stations (nst)')
        plt.ylabel('Frequency')
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.show()

    def dmin_histogram(self, figsize: tuple = (8, 5)) -> None:
        """Public method: Plot histogram of minimum distance to epicentre.

        Args:
            figsize (tuple): Figure size as (width, height). Default (8, 5).
        """
        if not HAS_MATPLOTLIB:
            print("Matplotlib not available for plotting.")
            return

        if 'dmin' not in self._df.columns:
            print("Column 'dmin' not found in dataframe.")
            return

        dmin_data = self._df['dmin'].dropna()
        plt.figure(figsize=figsize)
        plt.hist(dmin_data, bins=25, color='orange', edgecolor='black', alpha=0.7)
        plt.title('Distribution of Minimum Distance to Epicentre', fontsize=14)
        plt.xlabel('Distance (degrees)')
        plt.ylabel('Frequency')
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.show()

    def temporal_frequency(self, figsize: tuple = (14, 6)) -> None:
        """Public method: Plot temporal patterns (daily and hourly frequencies).

        Args:
            figsize (tuple): Figure size as (width, height). Default (14, 6).
        """
        if not HAS_MATPLOTLIB:
            print("Matplotlib not available for plotting.")
            return

        if 'hour' not in self._df.columns or 'day_name' not in self._df.columns:
            print("Columns 'hour' and/or 'day_name' not found. Ensure features are engineered.")
            return

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Hourly distribution
        hour_counts = self._df['hour'].value_counts().sort_index()
        axes[0].bar(hour_counts.index, hour_counts.values, color='purple', alpha=0.7, edgecolor='black')
        axes[0].set_title('Earthquakes by Hour of Day', fontsize=12)
        axes[0].set_xlabel('Hour (0-23)')
        axes[0].set_ylabel('Frequency')
        axes[0].grid(axis='y', linestyle='--', alpha=0.3)

        # Daily distribution
        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        day_counts = self._df['day_name'].value_counts().reindex(day_order, fill_value=0)
        axes[1].bar(range(len(day_order)), day_counts.values, color='cyan', alpha=0.7, edgecolor='black')
        axes[1].set_title('Earthquakes by Day of Week', fontsize=12)
        axes[1].set_xlabel('Day of Week')
        axes[1].set_ylabel('Frequency')
        axes[1].set_xticks(range(len(day_order)))
        axes[1].set_xticklabels(day_order, rotation=45, ha='right')
        axes[1].grid(axis='y', linestyle='--', alpha=0.3)

        plt.tight_layout()
        plt.show()

    def mag_by_depth_category_kde(self, figsize: tuple = (10, 6)) -> None:
        """Public method: Plot KDE of magnitude by depth category (for engineered features).

        Args:
            figsize (tuple): Figure size as (width, height). Default (10, 6).
        """
        if not HAS_MATPLOTLIB or not HAS_SCIPY:
            print("Matplotlib and SciPy required for KDE plotting.")
            return

        if 'depth_category' not in self._df.columns or 'mag' not in self._df.columns:
            print("Columns 'depth_category' and/or 'mag' not found. Ensure features are engineered.")
            return

        plt.figure(figsize=figsize)
        depth_categories = self._df['depth_category'].dropna().unique()
        colors = ['red', 'orange', 'blue']

        for i, cat in enumerate(sorted(depth_categories)):
            mag_subset = self._df[self._df['depth_category'] == cat]['mag'].dropna()
            if len(mag_subset) > 1:
                mag_subset.plot(kind='density', label=f'{cat}', color=colors[i % len(colors)], linewidth=2)

        plt.title('Magnitude Distribution by Depth Category (KDE)', fontsize=14)
        plt.xlabel('Magnitude')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.show()

    def magnitude_magnitude_error_scatter(self, figsize: tuple = (8, 6)) -> None:
        """Public method: Plot relationship between magnitude and magnitude error.

        Args:
            figsize (tuple): Figure size as (width, height). Default (8, 6).
        """
        if not HAS_MATPLOTLIB:
            print("Matplotlib not available for plotting.")
            return

        if 'mag' not in self._df.columns or 'magError' not in self._df.columns:
            print("Columns 'mag' and/or 'magError' not found.")
            return

        valid_data = self._df[['mag', 'magError']].dropna()
        if len(valid_data) == 0:
            print("No data available for plotting magnitude vs magnitude error.")
            return

        plt.figure(figsize=figsize)
        plt.scatter(valid_data['mag'], valid_data['magError'], alpha=0.5, s=10, c='darkgreen')
        plt.title('Magnitude vs Magnitude Error', fontsize=14)
        plt.xlabel('Magnitude')
        plt.ylabel('Magnitude Error')
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.show()

    def depth_latitude_scatter(self, figsize: tuple = (8, 6)) -> None:
        """Public method: Plot relationship between depth and latitude.

        Args:
            figsize (tuple): Figure size as (width, height). Default (8, 6).
        """
        if not HAS_MATPLOTLIB:
            print("Matplotlib not available for plotting.")
            return

        if 'depth' not in self._df.columns or 'latitude' not in self._df.columns:
            print("Columns 'depth' and/or 'latitude' not found.")
            return

        plt.figure(figsize=figsize)
        plt.scatter(self._df['latitude'], self._df['depth'], alpha=0.5, s=10, c='darkred')
        plt.title('Depth vs Latitude', fontsize=14)
        plt.xlabel('Latitude (Degrees)')
        plt.ylabel('Depth (km)')
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.show()


__all__ = [
    "libs",
    "apply_default_plot_style",
    "silence_warnings",
    "availability",
    "PROJECT_ROOT",
    "DATA_DIR",
    "OUTPUTS_DIR",
    "EarthquakeDataset",
    "EarthquakeVisualizer",
]
