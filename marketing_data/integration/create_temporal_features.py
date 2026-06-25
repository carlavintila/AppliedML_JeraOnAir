import pandas as pd
import numpy as np

input_file = "master_calendar_with_unified_marketing_2022_2026.csv"
output_file = "master_calendar_with_temporal_features_2022_2026_FIXED.csv"

df = pd.read_csv(input_file)

print("=" * 80)
print("STEP 4 FIXED: TEMPORAL FEATURE ENGINEERING")
print("=" * 80)

df["sale_date"] = pd.to_datetime(df["sale_date"], errors="coerce")

if df["sale_date"].isna().sum() > 0:
    raise ValueError("Bad sale_date values found.")

required_cols = [
    "sale_date",
    "festival_year",
    "tickets_sold",
    "marketing_total_posts",
    "marketing_total_reach",
    "marketing_total_engagement",
    "marketing_total_views_core",
    "marketing_total_comments",
    "marketing_total_shares",
    "marketing_both_platforms_active"
]

missing_cols = [col for col in required_cols if col not in df.columns]

if missing_cols:
    raise ValueError("Missing required columns: " + str(missing_cols))

df = df.sort_values(["festival_year", "sale_date"]).reset_index(drop=True)

lag_features = [
    "tickets_sold",
    "marketing_total_posts",
    "marketing_total_reach",
    "marketing_total_engagement",
    "marketing_total_views_core",
    "marketing_total_comments",
    "marketing_total_shares",
    "marketing_both_platforms_active"
]

lag_days = [1, 3, 7]

rolling_features = [
    "tickets_sold",
    "marketing_total_posts",
    "marketing_total_reach",
    "marketing_total_engagement",
    "marketing_total_views_core",
    "marketing_total_comments",
    "marketing_total_shares"
]

rolling_windows = [3, 7]

for col in set(lag_features + rolling_features):
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

# LAG FEATURES

print("\nCreating lag features...")

for col in lag_features:
    for lag in lag_days:
        df[f"{col}_lag_{lag}"] = (
            df.groupby("festival_year")[col].shift(lag)
        )

# FIXED ROLLING FEATURES

print("Creating leakage-safe rolling features...")

for col in rolling_features:
    for window in rolling_windows:
        new_col = f"{col}_roll_{window}_prior"

        df[new_col] = (
            df.groupby("festival_year")[col]
            .transform(lambda s: s.shift(1).rolling(window=window, min_periods=1).sum())
        )

# MOMENTUM FEATURES

print("Creating momentum features...")

momentum_base_features = [
    "tickets_sold",
    "marketing_total_posts",
    "marketing_total_reach",
    "marketing_total_engagement"
]

for col in momentum_base_features:
    roll_3 = f"{col}_roll_3_prior"
    roll_7 = f"{col}_roll_7_prior"

    df[f"{col}_momentum_3_vs_7"] = np.where(
        df[roll_7] > 0,
        df[roll_3] / df[roll_7],
        0
    )

# FILL START-OF-CYCLE NaNs

temporal_cols = [
    col for col in df.columns
    if "_lag_" in col or "_roll_" in col or "_momentum_" in col
]

df[temporal_cols] = df[temporal_cols].fillna(0)

# VALIDATION

print("\n" + "=" * 80)
print("VALIDATION")
print("=" * 80)

print("Output shape:", df.shape)
print("New temporal columns created:", len(temporal_cols))
print("Duplicate dates:", df["sale_date"].duplicated().sum())
print("Missing dates:", df["sale_date"].isna().sum())

print("\nFirst rows per festival year:")
for year in sorted(df["festival_year"].unique()):
    temp = df[df["festival_year"] == year].head(3)
    print("\nFestival year:", year)
    print(temp[[
        "sale_date",
        "tickets_sold",
        "tickets_sold_lag_1",
        "tickets_sold_roll_3_prior",
        "tickets_sold_roll_7_prior"
    ]])

# Important leakage check
first_rows = df.groupby("festival_year").head(1)

leak_check_cols = [
    "tickets_sold_lag_1",
    "tickets_sold_roll_3_prior",
    "tickets_sold_roll_7_prior"
]

print("\nStart-of-cycle leakage check:")
print(first_rows[["festival_year", "sale_date"] + leak_check_cols])

if (first_rows[leak_check_cols] != 0).any().any():
    raise ValueError("Leakage detected: first row of a festival cycle has past-history values.")

print("Start-of-cycle leakage check: PASSED")

# Null check
nulls = df[temporal_cols].isna().sum()
nulls = nulls[nulls > 0]

if len(nulls) == 0:
    print("No nulls in temporal columns.")
else:
    print(nulls)

# Negative check
negative_cols = {}

for col in temporal_cols:
    count = int((df[col] < 0).sum())
    if count > 0:
        negative_cols[col] = count

if len(negative_cols) == 0:
    print("No negative values in temporal columns.")
else:
    print("Negative values found:")
    print(negative_cols)

# SAVE FILE

df["sale_date"] = df["sale_date"].dt.strftime("%Y-%m-%d")

df.to_csv(output_file, index=False)

print("\n" + "=" * 80)
print("FILE SAVED")
print("=" * 80)
print("Saved as:", output_file)

print("\nStep 4 fixed complete.")
