import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_CSV = ROOT / "data" / "processed" / "josaa_iit_consolidated.csv"
NIRF_CSV = ROOT / "data" / "nirf_rankings.csv"
OUT_CSV = ROOT / "data" / "processed" / "features.csv"

def load_base():
    df = pd.read_csv(RAW_CSV)
    nirf = pd.read_csv(NIRF_CSV)
    return df, nirf

def add_nirf(df, nirf):
    df = df.merge(nirf, on="Institute", how="left")
    return df

def add_demand_score(df):
    df["Closing_Rank_Percentile"] = df.groupby(
        ["Year", "Seat_Type", "Gender"]
    )["Closing_Rank"].rank(pct=True)
    df["Demand_Score"] = 1 - df["Closing_Rank_Percentile"]
    return df

def add_desirability(df, nirf_weight=0.5, demand_weight=0.5):
    max_rank = df["NIRF_Engineering_Rank_2025"].max()
    df["NIRF_Score"] = 1 - ((df["NIRF_Engineering_Rank_2025"] - 1) / (max_rank - 1))
    df["Desirability_Score"] = (
        nirf_weight * df["NIRF_Score"] + demand_weight * df["Demand_Score"]
    )
    return df

def build_and_save():
    df, nirf = load_base()
    df = add_nirf(df, nirf)
    df = add_demand_score(df)
    df = add_desirability(df)
    df.to_csv(OUT_CSV, index=False)
    return df

if __name__ == "__main__":
    out = build_and_save()
    print("wrote", OUT_CSV, out.shape)
