"""
Trend Forecaster.

For every (Institute, Branch, Seat_Type, Gender) combo, fit a simple linear
regression of Closing_Rank (final round only, one point per year) against
Year, across however many of the 2016-2024 years that combo has data for.
Project next year's likely closing rank with a +/- band.

Combos with >=4 years of history get a real OLS fit, with the band derived
from the fit's own residual spread. Combos with <4 years fall back to
"carry forward the last value" with a fixed +/-20% band, flagged
method="carry_forward" so the UI can say "trend not established" instead of
presenting a regression line drawn through noise.
"""
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.linear_model import LinearRegression

ROOT = Path(__file__).resolve().parents[2]
RAW_CSV = ROOT / "data" / "processed" / "josaa_iit_consolidated.csv"
ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)

MIN_YEARS_FOR_TREND = 4
CARRY_FORWARD_BAND_PCT = 0.20


def final_round_series():
    df = pd.read_csv(RAW_CSV)
    final = df.loc[df.groupby(["Year", "Institute", "Branch", "Seat_Type", "Gender"])
                    ["Round"].transform("max") == df["Round"]]
    final = final.drop_duplicates(subset=["Year", "Institute", "Branch", "Seat_Type", "Gender"])
    return final[["Year", "Institute", "Branch", "Seat_Type", "Gender", "Closing_Rank"]]


def fit_one(years, ranks):
    years = np.array(years, dtype=float)
    ranks = np.array(ranks, dtype=float)
    next_year = int(years.max()) + 1

    if len(years) < MIN_YEARS_FOR_TREND:
        last_rank = ranks[years.argmax()]
        band = max(1, last_rank * CARRY_FORWARD_BAND_PCT)
        return {
            "n_years": len(years),
            "last_year": int(years.max()),
            "last_closing_rank": float(last_rank),
            "next_year": next_year,
            "predicted_closing_rank": float(last_rank),
            "lower_band": max(1, float(last_rank - band)),
            "upper_band": float(last_rank + band),
            "slope_per_year": 0.0,
            "r_squared": None,
            "method": "carry_forward",
        }

    X = years.reshape(-1, 1)
    model = LinearRegression().fit(X, ranks)
    pred = model.predict(X)
    resid_std = float(np.std(ranks - pred))
    band = max(1.0, 1.96 * resid_std)
    next_pred = float(model.predict([[next_year]])[0])
    ss_res = np.sum((ranks - pred) ** 2)
    ss_tot = np.sum((ranks - ranks.mean()) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else None

    return {
        "n_years": len(years),
        "last_year": int(years.max()),
        "last_closing_rank": float(ranks[years.argmax()]),
        "next_year": next_year,
        "predicted_closing_rank": max(1.0, next_pred),
        "lower_band": max(1.0, next_pred - band),
        "upper_band": next_pred + band,
        "slope_per_year": float(model.coef_[0]),
        "r_squared": r2,
        "method": "linear_regression",
    }


def train_and_save():
    series = final_round_series()
    rows = []
    for key, g in series.groupby(["Institute", "Branch", "Seat_Type", "Gender"]):
        fit = fit_one(g["Year"].tolist(), g["Closing_Rank"].tolist())
        institute, branch, seat_type, gender = key
        rows.append({"Institute": institute, "Branch": branch,
                      "Seat_Type": seat_type, "Gender": gender, **fit})

    forecast = pd.DataFrame(rows)
    forecast.to_csv(ARTIFACTS / "forecaster_table.csv", index=False)
    joblib.dump(forecast, ARTIFACTS / "forecaster_table.joblib")
    return forecast


if __name__ == "__main__":
    forecast = train_and_save()
    trend_n = (forecast["method"] == "linear_regression").sum()
    carry_n = (forecast["method"] == "carry_forward").sum()
    print(f"forecasted combos: {len(forecast)} (trend-fit: {trend_n}, carry-forward: {carry_n})")
