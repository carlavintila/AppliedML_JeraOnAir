import pandas as pd
import numpy as np

# STEP 2A:
# MERGE INSTAGRAM FEATURES ONTO MASTER OPERATIONAL CALENDAR

master_file = "master_operational_calendar_continuous_2022_2026.csv"

instagram_file = "instagram_daily_features_full_2019_2026.csv"

output_file = "master_calendar_with_instagram_2022_2026.csv"

# LOAD FILES

master = pd.read_csv(master_file)

ig = pd.read_csv(instagram_file)

print("=" * 80)
print("STEP 2A: MERGE INSTAGRAM ONTO MASTER OPERATIONAL CALENDAR")
print("=" * 80)

print("\nMaster calendar shape:", master.shape)
print("Instagram daily shape:", ig.shape)

# VALIDATE REQUIRED DATE COLUMNS

if "sale_date" not in master.columns:
    raise ValueError("sale_date missing in master calendar")

if "sale_date" not in ig.columns:
    raise ValueError("sale_date missing in Instagram file")

# PARSE DATES

master["sale_date"] = pd.to_datetime(
    master["sale_date"],
    errors="coerce"
)

ig["sale_date"] = pd.to_datetime(
    ig["sale_date"],
    errors="coerce"
)

master_bad_dates = master["sale_date"].isna().sum()
ig_bad_dates = ig["sale_date"].isna().sum()

print("\nBad master dates:", master_bad_dates)
print("Bad Instagram dates:", ig_bad_dates)

if master_bad_dates > 0:
    raise ValueError("Master calendar contains bad dates.")

if ig_bad_dates > 0:
    raise ValueError("Instagram file contains bad dates.")

# DUPLICATE DATE VALIDATION

master_duplicates = master["sale_date"].duplicated().sum()

ig_duplicates = ig["sale_date"].duplicated().sum()

print("\nDuplicate master dates:", master_duplicates)
print("Duplicate Instagram dates:", ig_duplicates)

if master_duplicates > 0:
    raise ValueError("Master calendar contains duplicate dates.")

if ig_duplicates > 0:
    raise ValueError("Instagram file contains duplicate dates.")

# INSPECT DATE COVERAGE

master_dates = set(master["sale_date"].dt.date)

ig_dates = set(ig["sale_date"].dt.date)

ig_inside_operational = len(master_dates.intersection(ig_dates))

ig_outside_operational = len(ig_dates - master_dates)

print("\nInstagram dates inside operational windows:", ig_inside_operational)
print("Instagram dates outside operational windows:", ig_outside_operational)

# IDENTIFY INSTAGRAM FEATURE COLUMNS

ig_feature_columns = [
    col for col in ig.columns
    if col != "sale_date"
]

print("\nInstagram feature columns found:", len(ig_feature_columns))

# VALIDATE NUMERIC FEATURE TYPES

numeric_ig_columns = []

non_numeric_ig_columns = []

for col in ig_feature_columns:

    if pd.api.types.is_numeric_dtype(ig[col]):
        numeric_ig_columns.append(col)
    else:
        non_numeric_ig_columns.append(col)

print("Numeric Instagram columns:", len(numeric_ig_columns))
print("Non-numeric Instagram columns:", len(non_numeric_ig_columns))

if len(non_numeric_ig_columns) > 0:

    print("\nNon-numeric Instagram columns:")
    print(non_numeric_ig_columns)

# MERGE INSTAGRAM ONTO MASTER CALENDAR

merged = master.merge(
    ig,
    on="sale_date",
    how="left"
)

print("\nMerged shape:", merged.shape)

# CREATE INSTAGRAM ACTIVITY FLAG

if "ig_posts_count" in merged.columns:

    merged["ig_activity_day"] = np.where(
        merged["ig_posts_count"].fillna(0) > 0,
        1,
        0
    )

else:
    merged["ig_activity_day"] = 0

# FILL OPERATIONAL NO-ACTIVITY DAYS WITH ZEROS

for col in numeric_ig_columns:

    merged[col] = merged[col].fillna(0)

# VALIDATION AFTER MERGE

print("\n" + "=" * 80)
print("VALIDATION AFTER INSTAGRAM MERGE")
print("=" * 80)

print("Final merged shape:", merged.shape)

print("\nDuplicate dates after merge:",
      merged["sale_date"].duplicated().sum())

print("Missing sale_date values:",
      merged["sale_date"].isna().sum())

if merged["sale_date"].duplicated().sum() > 0:
    raise ValueError("Duplicate dates created after Instagram merge.")

# VALIDATE ROW PRESERVATION

original_master_rows = len(master)

merged_rows = len(merged)

print("\nOriginal master rows:", original_master_rows)
print("Merged rows:", merged_rows)

if original_master_rows == merged_rows:
    print("Master calendar row preservation: PASSED")
else:
    raise ValueError("Master calendar row count changed after merge.")

# VALIDATE INSTAGRAM TOTALS PRESERVED

print("\n" + "=" * 80)
print("INSTAGRAM FEATURE TOTAL VALIDATION")
print("=" * 80)

important_validation_columns = []

possible_validation_cols = [
    "ig_posts_count",
    "ig_views_sum",
    "ig_reach_sum",
    "ig_likes_sum",
    "ig_comments_sum",
    "ig_shares_sum",
    "ig_saves_sum"
]

for col in possible_validation_cols:

    if col in ig.columns:
        important_validation_columns.append(col)

for col in important_validation_columns:

    original_total = ig[col].sum()

    merged_total = merged[col].sum()

    difference = original_total - merged_total

    print(
        f"{col} | original: {original_total} | "
        f"merged: {merged_total} | "
        f"difference: {difference}"
    )

# INSPECT OPERATIONAL NO-ACTIVITY DAYS

print("\n" + "=" * 80)
print("INSTAGRAM OPERATIONAL ACTIVITY ANALYSIS")
print("=" * 80)

print("Operational days with Instagram activity:",
      merged["ig_activity_day"].sum())

print("Operational days with NO Instagram activity:",
      len(merged) - merged["ig_activity_day"].sum())

# FINAL NULL INSPECTION

print("\n" + "=" * 80)
print("FINAL NULL INSPECTION")
print("=" * 80)

remaining_nulls = merged.isna().sum()

remaining_nulls = remaining_nulls[remaining_nulls > 0]

if len(remaining_nulls) == 0:
    print("No remaining null values.")
else:
    print(remaining_nulls)

# PREVIEW OUTPUT

print("\n" + "=" * 80)
print("PREVIEW OF MERGED DATASET")
print("=" * 80)

print(merged.head())

print("\nFinal column count:", len(merged.columns))

# SAVE FILE

merged["sale_date"] = merged["sale_date"].dt.strftime("%Y-%m-%d")

merged.to_csv(output_file, index=False)

print("\n" + "=" * 80)
print("MASTER + INSTAGRAM DATASET SAVED")
print("=" * 80)

print("Saved as:", output_file)

print("\nStep 2A complete.")
print("Next step: merge Facebook daily features onto this merged dataset.")
