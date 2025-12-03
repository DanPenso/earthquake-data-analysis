"""Generate the Section 8.2 epicentre map as static PNG (plus HTML if Plotly is available)."""

from pathlib import Path

import setup_libs as libs
from pandas.api.types import is_datetime64_any_dtype

pd = libs.pd
np = libs.np
px = libs.px
plt = libs.plt


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Replicates the notebook cleaning step."""
    df = df.copy()

    original_rows = len(df)
    df = df.drop_duplicates()
    # Convert time fields
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df["updated"] = pd.to_datetime(df["updated"], errors="coerce")

    rows_before_id_clean = len(df)
    df = df.sort_values("updated").drop_duplicates(subset="id", keep="last")

    for col in ["latitude", "longitude", "depth", "mag"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    essential_cols = ["time", "latitude", "longitude", "depth", "mag"]
    df = df.dropna(subset=essential_cols)

    valid_lat = df["latitude"].between(-90, 90)
    valid_lon = df["longitude"].between(-180, 180)
    valid_depth = df["depth"].between(0, 700)
    valid_mag = df["mag"].between(0, 10)
    df = df[valid_lat & valid_lon & valid_depth & valid_mag]

    if "type" in df.columns:
        df = df[df["type"] == "earthquake"]

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering from the notebook."""
    df = df.copy()

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

    def hour_to_part(h):
        if h < 6:
            return "night"
        if h < 12:
            return "morning"
        if h < 18:
            return "afternoon"
        return "evening"

    df["part_of_day"] = df["hour"].apply(hour_to_part)

    def month_to_season(m):
        if m in (12, 1, 2):
            return "winter"
        if m in (3, 4, 5):
            return "spring"
        if m in (6, 7, 8):
            return "summer"
        return "autumn"

    df["season"] = df["month"].apply(month_to_season)

    df["depth_category"] = pd.cut(
        df["depth"],
        bins=[0, 70, 300, 700],
        labels=["shallow", "intermediate", "deep"],
        right=False,
    )
    df["mag_category"] = pd.cut(
        df["mag"],
        bins=[0, 3, 4, 5, 6, 7, 8, 10],
        labels=["minor", "light", "moderate", "strong", "major", "great", "massive"],
        right=False,
    )
    df["is_strong_quake"] = df["mag"] >= 6.0
    df["energy_log10_J"] = 1.5 * df["mag"] + 4.8

    df["abs_latitude"] = df["latitude"].abs()
    df["abs_longitude"] = df["longitude"].abs()
    df["distance_from_equator_km"] = df["abs_latitude"] * 111.0
    df["distance_from_prime_meridian_km"] = (
        df["abs_longitude"] * 111.0 * np.cos(np.deg2rad(df["latitude"]))
    )
    df["hemisphere_NS"] = np.where(df["latitude"] >= 0, "north", "south")
    df["hemisphere_EW"] = np.where(df["longitude"] >= 0, "east", "west")

    def classify_region(lon):
        if lon < -100:
            return "Americas_west"
        if lon < -30:
            return "Americas_east_Atlantic"
        if lon < 60:
            return "Europe_Africa"
        if lon < 150:
            return "Asia_WestPacific"
        return "Pacific_Oceania"

    df["broad_region"] = df["longitude"].apply(classify_region)

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

    cat_cols = [
        "depth_category",
        "mag_category",
        "hemisphere_NS",
        "hemisphere_EW",
        "broad_region",
        "part_of_day",
        "season",
    ]
    for c in cat_cols:
        if c in df.columns:
            df[f"{c}_code"] = df[c].astype("category").cat.codes

    return df


def prepare_map_df(csv_path: Path) -> pd.DataFrame:
    """Load, clean, and feature-engineer the map dataframe."""
    eq_df = pd.read_csv(csv_path)
    cleaned_eq_df = clean_data(eq_df)
    featured_eq_df = engineer_features(cleaned_eq_df)

    map_cols = ["latitude", "longitude", "mag", "mag_category", "broad_region", "depth", "place"]
    available_cols = [c for c in map_cols if c in featured_eq_df.columns]
    map_df = featured_eq_df[available_cols].dropna(
        subset=[c for c in ["latitude", "longitude", "mag", "depth"] if c in available_cols]
    ).copy()
    if map_df.empty:
        raise ValueError("No latitude/longitude samples available for Section 8.2.")

    map_df["mag_size"] = np.clip(map_df["mag"], 3, None)
    map_df["is_strong"] = map_df["mag"] >= 6.0
    map_df["hover_label"] = (
        map_df["place"].fillna("Unknown location")
        + " | Mw " + map_df["mag"].round(1).astype(str)
        + " | depth " + map_df["depth"].round(0).astype(int).astype(str) + " km"
    )
    return map_df


def build_region_summary(map_df: pd.DataFrame) -> pd.DataFrame:
    region_summary = (
        map_df.assign(strong_pct=map_df["is_strong"].astype(float))
        .groupby("broad_region")
        .agg(
            events=("mag", "size"),
            pct_global=("mag", lambda s: 100 * len(s) / len(map_df)),
            pct_strong=("strong_pct", "mean"),
            median_depth=("depth", "median"),
        )
        .sort_values("events", ascending=False)
    )
    region_summary["pct_strong"] = region_summary["pct_strong"] * 100
    region_summary = region_summary.round({"pct_global": 1, "pct_strong": 1, "median_depth": 0})
    return region_summary


def build_plotly_map(map_df: pd.DataFrame):
    if px is None:
        return None
    fig = px.scatter_geo(
        map_df,
        lat="latitude",
        lon="longitude",
        color="broad_region",
        size="mag_size",
        size_max=18,
        hover_name="hover_label",
        hover_data={"mag": ":.1f", "depth": ":.0f", "mag_category": True, "broad_region": True},
        projection="natural earth",
        template="plotly_white",
        title="Global epicentre distribution (bubble size ~ magnitude)",
    )
    fig.update_layout(legend_title_text="Broad region")
    return fig


def save_matplotlib_png(map_df: pd.DataFrame, png_path: Path):
    if plt is None:
        raise RuntimeError("Matplotlib is not available for static export.")
    fig, ax = plt.subplots(figsize=(14, 7))
    scatter = ax.scatter(
        map_df["longitude"],
        map_df["latitude"],
        c=map_df["mag"],
        cmap="plasma",
        s=np.square(map_df["mag_size"]) * 1.5,
        alpha=0.6,
        linewidths=0.15,
        edgecolor="black",
    )
    ax.set_title("Global epicentre distribution (bubble size ~ magnitude)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.colorbar(scatter, ax=ax, label="Magnitude (Mw)")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    root = Path(__file__).parent
    csv_path = root / "earthquake_dataset.csv"
    png_path = root / "epicentre_map_section8_2.png"
    html_path = root / "epicentre_map_section8_2.html"
    summary_path = root / "region_summary_section8_2.csv"

    map_df = prepare_map_df(csv_path)
    region_summary = build_region_summary(map_df)

    if px is not None:
        fig = build_plotly_map(map_df)
        fig.write_html(html_path, include_plotlyjs="cdn")
        print(f"Wrote interactive HTML map to {html_path}")
    else:
        print("Plotly not available; skipped HTML export.")

    save_matplotlib_png(map_df, png_path)
    print(f"Wrote static PNG map to {png_path}")

    region_summary.to_csv(summary_path)
    print(f"Wrote region summary to {summary_path}")


if __name__ == "__main__":
    main()
