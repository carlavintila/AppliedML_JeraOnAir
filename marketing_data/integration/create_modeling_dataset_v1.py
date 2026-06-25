import pandas as pd
import numpy as np

# =========================================================
# MERGE GROUPMATE EXTRA FEATURES INTO OUR MODELING DATASET
# =========================================================

our_file = "modeling_dataset_v1.csv"
groupmate_file = "ticket_sales_12_41_22_05.csv"

output_file = "modeling_dataset_v2_with_groupmate_features.csv"
report_file = "groupmate_feature_merge_report.csv"

# =========================================================
# LOAD FILES
# =========================================================

ours = pd.read_csv(our_file)
group = pd.read_csv(groupmate_file)

print("=" * 80)
print("MERGE GROUPMATE EXTRA FEATURES INTO OUR MODELING DATASET")
print("=" * 80)

print("\nOur dataset shape:", ours.shape)
print("Groupmate dataset shape:", group.shape)

# =========================================================
# BASIC DATE VALIDATION
# =========================================================

ours["sale_date"] = pd.to_datetime(ours["sale_date"], errors="coerce")
group["sale_date"] = pd.to_datetime(group["sale_date"], errors="coerce")

if ours["sale_date"].isna().sum() > 0:
    raise ValueError("Bad sale_date values in our dataset.")

if group["sale_date"].isna().sum() > 0:
    raise ValueError("Bad sale_date values in groupmate dataset.")

if ours["sale_date"].duplicated().sum() > 0:
    raise ValueError("Duplicate sale_date values in our dataset.")

print("\nDate validation: PASSED")

# =========================================================
# DO NOT IMPORT THESE FROM GROUPMATE FILE
# =========================================================
# Reason:
# These are already handled better in our leakage-safe pipeline.

do_not_import_prefixes = [
    "ig_",
    "sales_lag_",
    "sales_roll_",
]

do_not_import_exact = [
    "sale_date",
    "festival_year",
    "days_to_event",
    "tickets_sold",
    "is_event_day",
    "cumulative_sales_prior",
    "ig_video_posts_count",
    "ig_other_posts_count",
]

# =========================================================
# ADD CALENDAR FEATURES FROM OUR OWN MASTER TIMELINE
# =========================================================
# Better than importing them, because our file has the correct continuous calendar.

ours["month"] = ours["sale_date"].dt.month
ours["weekday"] = ours["sale_date"].dt.weekday
ours["weekofyear"] = ours["sale_date"].dt.isocalendar().week.astype(int)
ours["is_weekend"] = ours["weekday"].isin([5, 6]).astype(int)

# Sales-open features derived from our own operational calendar
sales_open = (
    ours.groupby("festival_year")["sale_date"]
    .min()
    .reset_index()
    .rename(columns={"sale_date": "sales_open_date"})
)

ours = ours.merge(sales_open, on="festival_year", how="left")

ours["days_since_sales_open"] = (
    ours["sale_date"] - ours["sales_open_date"]
).dt.days

# Keep sales_open_date as traceability string
ours["sales_open_date"] = ours["sales_open_date"].dt.strftime("%Y-%m-%d")

print("\nCalendar and sales-open features created from our own timeline.")

# =========================================================
# SELECT GROUPMATE ARTIST / LINEUP FEATURES ONLY
# =========================================================

candidate_groupmate_features = [
    "avg_popularity (0-100)",
    "avg_popularity_spotify (0-100)",
    "num_headliner",
    "headliner_popularity (0-100)",
    "headliner_popularity_spotify (0-100)",
    "num_headliner_2",
    "headliner_2_popularity (0-100)",
    "headliner_2_popularity_spotify (0-100)",
    "num_headliner_combined",
    "headliner_combined_popularity (0-100)",
    "headliner_combined_popularity_spotify (0-100)",
    "genre_entropy (0-1)",
    "num_genres",
    "dominant_genre_ratio (0-1)",
]

available_groupmate_features = [
    col for col in candidate_groupmate_features
    if col in group.columns
]

missing_groupmate_features = [
    col for col in candidate_groupmate_features
    if col not in group.columns
]

print("\nAvailable groupmate features to import:")
for col in available_groupmate_features:
    print("-", col)

if missing_groupmate_features:
    print("\nMissing expected groupmate features:")
    for col in missing_groupmate_features:
        print("-", col)

# =========================================================
# CHECK IF ARTIST FEATURES ARE STABLE PER FESTIVAL YEAR
# =========================================================

print("\n" + "=" * 80)
print("FESTIVAL-YEAR STABILITY CHECK FOR GROUPMATE FEATURES")
print("=" * 80)

stability_rows = []

for col in available_groupmate_features:
    max_unique_per_year = group.groupby("festival_year")[col].nunique(dropna=False).max()

    stability_rows.append({
        "feature": col,
        "max_unique_values_within_festival_year": max_unique_per_year
    })

stability_report = pd.DataFrame(stability_rows)

print(stability_report)

unstable_features = stability_report[
    stability_report["max_unique_values_within_festival_year"] > 1
]["feature"].tolist()

if len(unstable_features) > 0:
    print("\nWARNING: These features vary within festival_year:")
    for col in unstable_features:
        print("-", col)
    print("\nWe will still merge by festival_year using the first value, but inspect later.")
else:
    print("\nAll selected groupmate features are stable within festival_year.")

# =========================================================
# CREATE FESTIVAL-YEAR LEVEL GROUPMATE FEATURE TABLE
# =========================================================

group_year_features = (
    group[["festival_year"] + available_groupmate_features]
    .sort_values("festival_year")
    .groupby("festival_year", as_index=False)
    .first()
)

# Rename columns to cleaner names
rename_map = {
    "avg_popularity (0-100)": "artist_avg_popularity_0_100",
    "avg_popularity_spotify (0-100)": "artist_avg_popularity_spotify_0_100",
    "num_headliner": "artist_num_headliner",
    "headliner_popularity (0-100)": "artist_headliner_popularity_0_100",
    "headliner_popularity_spotify (0-100)": "artist_headliner_popularity_spotify_0_100",
    "num_headliner_2": "artist_num_headliner_2",
    "headliner_2_popularity (0-100)": "artist_headliner_2_popularity_0_100",
    "headliner_2_popularity_spotify (0-100)": "artist_headliner_2_popularity_spotify_0_100",
    "num_headliner_combined": "artist_num_headliner_combined",
    "headliner_combined_popularity (0-100)": "artist_headliner_combined_popularity_0_100",
    "headliner_combined_popularity_spotify (0-100)": "artist_headliner_combined_popularity_spotify_0_100",
    "genre_entropy (0-1)": "artist_genre_entropy_0_1",
    "num_genres": "artist_num_genres",
    "dominant_genre_ratio (0-1)": "artist_dominant_genre_ratio_0_1",
}

group_year_features = group_year_features.rename(columns=rename_map)

# =========================================================
# MERGE INTO OUR DATASET
# =========================================================

before_shape = ours.shape

merged = ours.merge(
    group_year_features,
    on="festival_year",
    how="left"
)

print("\nShape before merge:", before_shape)
print("Shape after merge:", merged.shape)

# =========================================================
# VALIDATION
# =========================================================

print("\n" + "=" * 80)
print("VALIDATION")
print("=" * 80)

print("Rows before:", len(ours))
print("Rows after:", len(merged))

if len(ours) != len(merged):
    raise ValueError("Row count changed after merge.")

print("Row preservation: PASSED")

print("Duplicate sale_date:", merged["sale_date"].duplicated().sum())
print("Missing sale_date:", merged["sale_date"].isna().sum())

if merged["sale_date"].duplicated().sum() > 0:
    raise ValueError("Duplicate sale_date created.")

# Target preservation
print("\nOriginal tickets_sold total:", ours["tickets_sold"].sum())
print("Merged tickets_sold total:", merged["tickets_sold"].sum())

if ours["tickets_sold"].sum() != merged["tickets_sold"].sum():
    raise ValueError("tickets_sold total changed after merge.")

print("Target preservation: PASSED")

# Check new missing values
new_columns = [col for col in merged.columns if col not in df.columns] if False else []

new_added_cols = [
    "month",
    "weekday",
    "weekofyear",
    "is_weekend",
    "sales_open_date",
    "days_since_sales_open"
] + [rename_map[col] for col in available_groupmate_features if col in rename_map]

missing_new = merged[new_added_cols].isna().sum()
missing_new = missing_new[missing_new > 0]

print("\nMissing values in newly added columns:")
if len(missing_new) == 0:
    print("No missing values in newly added columns.")
else:
    print(missing_new)

# =========================================================
# SAVE MERGE REPORT
# =========================================================

report_rows = []

for col in new_added_cols:
    report_rows.append({
        "added_column": col,
        "missing_count": int(merged[col].isna().sum()),
        "unique_values": int(merged[col].nunique(dropna=False)),
        "dtype": str(merged[col].dtype)
    })

report_df = pd.DataFrame(report_rows)
report_df.to_csv(report_file, index=False)

# =========================================================
# FINAL CLEAN SAVE
# =========================================================

merged["sale_date"] = merged["sale_date"].dt.strftime("%Y-%m-%d")

merged.to_csv(output_file, index=False)

print("\n" + "=" * 80)
print("FILES SAVED")
print("=" * 80)
print("Saved:", output_file)
print("Saved:", report_file)

print("\nFinal dataset shape:", merged.shape)
print("Feature enrichment complete.")