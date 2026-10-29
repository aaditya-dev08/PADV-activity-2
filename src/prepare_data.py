# coding: utf-8
"""
src/prepare_data.py
-------------------
Spotify Data Pipeline
Loads raw dataset via kagglehub, cleans & engineers features,
saves to data/spotify_cleaned.csv.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Resolve project root
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_OUT = os.path.join(ROOT, "data", "spotify_cleaned.csv")


def load_raw_data() -> pd.DataFrame:
    """Download dataset from KaggleHub and return as DataFrame."""
    print("[*] Loading dataset from KaggleHub ...")
    try:
        import kagglehub
        path = kagglehub.dataset_download("maharshipandya/-spotify-tracks-dataset")
        csv_path = os.path.join(path, "dataset.csv")
        if not os.path.exists(csv_path):
            # fallback: find any .csv in the downloaded folder
            csvs = [f for f in os.listdir(path) if f.endswith(".csv")]
            if not csvs:
                raise FileNotFoundError("No CSV found in {}".format(path))
            csv_path = os.path.join(path, csvs[0])
        df = pd.read_csv(csv_path, index_col=0)
        print("   [OK] Raw shape: {}".format(df.shape))
        return df
    except Exception as e:
        print("   [ERR] KaggleHub error: {}".format(e))
        print("   [TIP] Ensure KAGGLE_USERNAME and KAGGLE_KEY env vars are set.")
        sys.exit(1)


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows by track_id."""
    before = len(df)
    if "track_id" in df.columns:
        df = df.drop_duplicates(subset="track_id")
    else:
        df = df.drop_duplicates()
    after = len(df)
    print("   [-] Duplicates dropped: {:,} rows removed".format(before - after))
    return df


def drop_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with nulls in key columns."""
    key_cols = [
        "popularity", "danceability", "energy", "loudness",
        "valence", "tempo", "acousticness", "instrumentalness",
        "speechiness", "liveness", "track_genre",
    ]
    existing = [c for c in key_cols if c in df.columns]
    before = len(df)
    df = df.dropna(subset=existing)
    print("   [-] Nulls dropped: {:,} rows removed".format(before - len(df)))
    return df


def cap_outliers_iqr(df: pd.DataFrame) -> pd.DataFrame:
    """Cap outliers via IQR for tempo, duration_ms, loudness."""
    cols = ["tempo", "duration_ms", "loudness"]
    for col in cols:
        if col not in df.columns:
            continue
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        before_min, before_max = df[col].min(), df[col].max()
        df[col] = df[col].clip(lower=lower, upper=upper)
        print("   [IQR] cap [{}]: [{:.1f}, {:.1f}] -> [{:.1f}, {:.1f}]".format(
            col, before_min, before_max, lower, upper))
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create derived columns."""
    # duration_min
    if "duration_ms" in df.columns:
        df["duration_min"] = df["duration_ms"] / 60_000

    # popularity_segment
    if "popularity" in df.columns:
        df["popularity_segment"] = pd.cut(
            df["popularity"],
            bins=[-1, 33, 66, 100],
            labels=["Low", "Medium", "High"],
        )

    # is_hit  (target variable)
    if "popularity" in df.columns:
        df["is_hit"] = (df["popularity"] > 50).astype(int)

    # energy_level
    if "energy" in df.columns:
        df["energy_level"] = pd.cut(
            df["energy"],
            bins=[-0.01, 0.33, 0.66, 1.01],
            labels=["Low", "Medium", "High"],
        )

    print("   [+] Features engineered: duration_min, popularity_segment, is_hit, energy_level")
    return df


def filter_duration(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only tracks <= 10 minutes."""
    before = len(df)
    if "duration_min" in df.columns:
        df = df[df["duration_min"] <= 10]
    after = len(df)
    print("   [~] Duration filter (<=10 min): {:,} rows removed".format(before - after))
    return df


def save(df: pd.DataFrame) -> None:
    os.makedirs(os.path.dirname(DATA_OUT), exist_ok=True)
    df.to_csv(DATA_OUT, index=False)
    print("\n[SAVED] Cleaned data -> {}".format(DATA_OUT))
    print("   Final shape : {}".format(df.shape))
    if "is_hit" in df.columns:
        hits = df["is_hit"].sum()
        pct = 100 * hits / len(df)
        print("   Hit tracks  : {:,} ({:.1f}%)".format(hits, pct))


def main():
    print("\n" + "=" * 50)
    print("  SPOTIFY DATA PIPELINE")
    print("=" * 50 + "\n")

    df = load_raw_data()
    df = drop_duplicates(df)
    df = drop_nulls(df)
    df = cap_outliers_iqr(df)
    df = engineer_features(df)
    df = filter_duration(df)
    save(df)

    print("\n" + "=" * 50)
    print("  COLUMN SUMMARY")
    print("=" * 50)
    print(df.dtypes.to_string())
    print("\nPipeline complete!\n")


if __name__ == "__main__":
    main()
