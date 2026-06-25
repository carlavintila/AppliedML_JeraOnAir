import pandas as pd
import numpy as np

# STEP 2B:
# MERGE FACEBOOK DAILY FEATURES ONTO MASTER + INSTAGRAM DATASET

master_file = "master_calendar_with_instagram_2022_2026.csv"
facebook_file = "facebook_daily_features_full_2019_2026.csv"

output_file = "master_calendar_with_instagram_facebook_2022_2026.csv"

# LOAD FILES

master = pd.read_csv(master_file)
fb = pd.read_csv(facebook_file)

print("=" * 80)
print("STEP 2B: MERGE FACEBOOK ONTO MASTER + INSTAGRAM DATASET")
print("=" * 80)

print("\nMaster + Instagram shape:", master.shape)
print("Facebook daily shape:", fb.shape)

# DATE COLUMN CHECKS

if "sale_date" not in master.columns:
    raise ValueError("sale_date column missing in master file.")

if "date" not in fb.columns:
    raise ValueError("date column missing in Facebook file.")

master["sale_date"] = pd.to_datetime(master["sale_date"], errors="coerce")
fb["date"] = pd.to_datetime(fb["date"], errors="coerce")

print("\nBad master dates:", master["sale_date"].isna().sum())
print("Bad Facebook dates:", fb["date"].isna().sum())

if master["sale_date"].isna().sum() > 0:
    raise ValueError("Bad dates found in master file.")

if fb["date"].isna().sum() > 0:
    raise ValueError("Bad dates found in Facebook file.")

print("\nDuplicate master dates:", master["sale_date"].duplicated().sum())
print("Duplicate Facebook dates:", fb["date"].duplicated().sum())

if master["sale_date"].duplicated().sum() > 0:
    raise ValueError("Duplicate dates found in master file.")

if fb["date"].duplicated().sum() > 0:
    raise ValueError("Duplicate dates found in Facebook file.")

# Rename Facebook date column for clean merge
fb = fb.rename(columns={"date": "sale_date"})

# DATE COVERAGE CHECK

master_dates = set(master["sale_date"].dt.date)
fb_dates = set(fb["sale_date"].dt.date)

fb_inside_operational = len(master_dates.intersection(fb_dates))
fb_outside_operational = len(fb_dates - master_dates)

print("\nFacebook dates inside operational windows:", fb_inside_operational)
print("Facebook dates outside operational windows:", fb_outside_operational)

# IDENTIFY FACEBOOK FEATURE GROUPS

fb_feature_columns = [col for col in fb.columns if col != "sale_date"]

print("\nFacebook feature columns found:", len(fb_feature_columns))

# These columns should preserve NaN because missing means historically unavailable
fb_meta_nan_preserve_columns = [
    "fb_total_views_sum",
    "fb_organic_views_sum",
    "fb_promoted_views_sum",
    "fb_unique_impressions_sum",
    "fb_negative_feedback_sum",
    "fb_unique_negative_feedback_sum",
    "fb_has_promoted_views_posts_count",
    "fb_total_views_data_available_posts",
    "fb_negative_feedback_data_available_posts",
    "fb_views_data_available",
    "fb_negative_feedback_data_available"
]

fb_meta_nan_preserve_columns = [
    col for col in fb_meta_nan_preserve_columns
    if col in fb.columns
]

# All other Facebook numeric columns can be zero-filled for no Facebook activity days
fb_zero_fill_columns = [
    col for col in fb_feature_columns
    if col not in fb_meta_nan_preserve_columns
]

print("\nFacebook zero-fill columns:", len(fb_zero_fill_columns))
print("Facebook NaN-preserve Meta columns:", len(fb_meta_nan_preserve_columns))

# MERGE FACEBOOK ONTO MASTER DATASET

merged = master.merge(
    fb,
    on="sale_date",
    how="left"
)

print("\nMerged shape before filling:", merged.shape)

# FILL FACEBOOK NO-ACTIVITY DAYS


for col in fb_zero_fill_columns:
    merged[col] = merged[col].fillna(0)

# For data availability flags:
# If there was no Facebook row that day, availability is 0.
# But for Meta sum columns, NaN preservation is handled separately.
for flag_col in [
    "fb_views_data_available",
    "fb_negative_feedback_data_available",
    "fb_total_views_data_available_posts",
    "fb_negative_feedback_data_available_posts"
]:
    if flag_col in merged.columns:
        merged[flag_col] = merged[flag_col].fillna(0)


print("\n" + "=" * 80)
print("VALIDATION AFTER FACEBOOK MERGE")
print("=" * 80)

print("Final merged shape:", merged.shape)

print("\nDuplicate dates after merge:",
      merged["sale_date"].duplicated().sum())

print("Missing sale_date values:",
      merged["sale_date"].isna().sum())

if merged["sale_date"].duplicated().sum() > 0:
    raise ValueError("Duplicate dates created after Facebook merge.")

# Row preservation
print("\nOriginal master rows:", len(master))
print("Merged rows:", len(merged))

if len(master) == len(merged):
    print("Master row preservation: PASSED")
else:
    raise ValueError("Master row count changed after Facebook merge.")


print("\n" + "=" * 80)
print("FACEBOOK FEATURE TOTAL VALIDATION")
print("=" * 80)

validation_cols = [
    "fb_posts_count",
    "fb_total_engagement_sum",
    "fb_reactions_sum",
    "fb_comments_sum",
    "fb_shares_sum",
    "fb_total_clicks_sum",
    "fb_link_clicks_sum",
    "fb_total_reach_sum",
    "fb_video_views_3s_sum"
]

for col in validation_cols:
    if col in fb.columns and col in merged.columns:
        original_total = fb[col].sum()
        merged_total = merged[col].sum()
        difference = original_total - merged_total

        print(
            f"{col} | original Facebook: {original_total} | "
            f"merged operational: {merged_total} | "
            f"excluded outside windows: {difference}"
        )

# FACEBOOK ACTIVITY ANALYSIS


print("\n" + "=" * 80)
print("FACEBOOK OPERATIONAL ACTIVITY ANALYSIS")
print("=" * 80)

if "fb_has_post" in merged.columns:
    print("Operational days with Facebook activity:",
          int(merged["fb_has_post"].sum()))

    print("Operational days with NO Facebook activity:",
          int(len(merged) - merged["fb_has_post"].sum()))

if "fb_posts_count" in merged.columns:
    print("Facebook posts inside operational windows:",
          int(merged["fb_posts_count"].sum()))

# META NaN PRESERVATION CHECK

print("\n" + "=" * 80)
print("FACEBOOK META NaN PRESERVATION CHECK")
print("=" * 80)

meta_check_cols = [
    "fb_total_views_sum",
    "fb_organic_views_sum",
    "fb_promoted_views_sum",
    "fb_negative_feedback_sum"
]

for col in meta_check_cols:
    if col in merged.columns:
        print(
            col,
            "| NaN rows:",
            merged[col].isna().sum(),
            "| non-NaN rows:",
            merged[col].notna().sum()
        )

# FINAL NULL INSPECTION

print("\n" + "=" * 80)
print("FINAL NULL INSPECTION")
print("=" * 80)

remaining_nulls = merged.isna().sum()
remaining_nulls = remaining_nulls[remaining_nulls > 0]

if len(remaining_nulls) == 0:
    print("No remaining null values.")
else:
    print("Remaining null values:")
    print(remaining_nulls)

# PREVIEW OUTPUT

print("\n" + "=" * 80)
print("PREVIEW OF FINAL MERGED DATASET")
print("=" * 80)

print(merged.head())

print("\nFinal column count:", len(merged.columns))

# SAVE FILE

merged["sale_date"] = merged["sale_date"].dt.strftime("%Y-%m-%d")

merged.to_csv(output_file, index=False)

print("\n" + "=" * 80)
print("MASTER + INSTAGRAM + FACEBOOK DATASET SAVED")
print("=" * 80)

print("Saved as:", output_file)

print("\nStep 2B complete.")
print("Next step: create unified online marketing features.")
