import pandas as pd
import numpy as np

# STEP 3C:
# ADD NEWER META FEATURES WITH SAFE NaN PRESERVATION

post_file = "facebook_posts_clean_all_years_sorted.csv"

step3b_file = "facebook_daily_features_step3B_categorical.csv"

output_file = "facebook_daily_features_full_2019_2026.csv"

# LOAD FILES

posts = pd.read_csv(post_file)

daily = pd.read_csv(step3b_file)

print("=" * 80)
print("STEP 3C: NEWER META FEATURES WITH NaN PRESERVATION")
print("=" * 80)

print("\nPost-level shape:", posts.shape)
print("Step 3B shape:", daily.shape)

# VALIDATE REQUIRED COLUMNS

required_meta_columns = [
    "publish_time",
    "permalink",
    "total_views",
    "organic_views",
    "promoted_views",
    "unique_impressions",
    "negative_feedback",
    "unique_negative_feedback",
    "has_promoted_views"
]

missing_cols = [
    col for col in required_meta_columns
    if col not in posts.columns
]

if len(missing_cols) > 0:
    raise ValueError(
        "Missing required Meta columns: " + str(missing_cols)
    )

print("\nRequired Meta column check passed.")

# CREATE DATE COLUMN

posts["publish_time_parsed"] = pd.to_datetime(
    posts["publish_time"],
    errors="coerce",
    utc=True
)

bad_dates = posts["publish_time_parsed"].isna().sum()

print("Bad publish_time values:", bad_dates)

if bad_dates > 0:
    raise ValueError("Bad publish_time values found.")

posts["date"] = posts["publish_time_parsed"].dt.date.astype(str)

daily["date"] = pd.to_datetime(
    daily["date"],
    errors="coerce"
).dt.date.astype(str)

# DEFINE NEWER META FEATURES
# Important:
# These features only exist in newer Meta exports.
# Missing historically does NOT mean zero.
# So we preserve NaN intelligently.

meta_feature_columns = [
    "total_views",
    "organic_views",
    "promoted_views",
    "unique_impressions",
    "negative_feedback",
    "unique_negative_feedback",
    "has_promoted_views"
]

# CONVERT TO NUMERIC

for col in meta_feature_columns:

    posts[col] = pd.to_numeric(
        posts[col],
        errors="coerce"
    )

# CREATE DAILY META FEATURES

daily_meta = posts.groupby("date").agg(

    # Core Meta features
    fb_total_views_sum=("total_views", "sum"),
    fb_organic_views_sum=("organic_views", "sum"),
    fb_promoted_views_sum=("promoted_views", "sum"),

    fb_unique_impressions_sum=("unique_impressions", "sum"),

    fb_negative_feedback_sum=("negative_feedback", "sum"),
    fb_unique_negative_feedback_sum=("unique_negative_feedback", "sum"),

    fb_has_promoted_views_posts_count=("has_promoted_views", "sum"),

    # Data availability counts
    fb_total_views_data_available_posts=("total_views", "count"),
    fb_negative_feedback_data_available_posts=("negative_feedback", "count")

).reset_index()

# PRESERVE HISTORICAL MISSINGNESS

# Total views group
view_columns = [
    "fb_total_views_sum",
    "fb_organic_views_sum",
    "fb_promoted_views_sum",
    "fb_unique_impressions_sum"
]

for col in view_columns:

    daily_meta[col] = np.where(
        daily_meta["fb_total_views_data_available_posts"] == 0,
        np.nan,
        daily_meta[col]
    )

# Negative feedback group
negative_feedback_columns = [
    "fb_negative_feedback_sum",
    "fb_unique_negative_feedback_sum",
    "fb_has_promoted_views_posts_count"
]

for col in negative_feedback_columns:

    daily_meta[col] = np.where(
        daily_meta["fb_negative_feedback_data_available_posts"] == 0,
        np.nan,
        daily_meta[col]
    )

# CREATE DATA AVAILABILITY FLAGS

daily_meta["fb_views_data_available"] = np.where(
    daily_meta["fb_total_views_data_available_posts"] > 0,
    1,
    0
)

daily_meta["fb_negative_feedback_data_available"] = np.where(
    daily_meta["fb_negative_feedback_data_available_posts"] > 0,
    1,
    0
)

# MERGE INTO STEP 3B DATASET

final_daily = daily.merge(
    daily_meta,
    on="date",
    how="left"
)

# VALIDATION

print("\n" + "=" * 80)
print("VALIDATION AFTER STEP 3C")
print("=" * 80)

print("Step 3B shape:", daily.shape)
print("Final Step 3C shape:", final_daily.shape)

print("Duplicate dates:", final_daily["date"].duplicated().sum())
print("Missing dates:", final_daily["date"].isna().sum())

if final_daily["date"].duplicated().sum() > 0:
    raise ValueError("Duplicate dates created after Step 3C merge.")

# VALIDATE HISTORICAL NaN PRESERVATION

print("\n" + "=" * 80)
print("HISTORICAL NaN PRESERVATION CHECK")
print("=" * 80)

meta_validation_columns = [
    "fb_total_views_sum",
    "fb_organic_views_sum",
    "fb_promoted_views_sum",
    "fb_negative_feedback_sum"
]

for col in meta_validation_columns:

    nan_count = final_daily[col].isna().sum()

    non_nan_count = final_daily[col].notna().sum()

    print(
        col,
        "| NaN days:",
        nan_count,
        "| Available-data days:",
        non_nan_count
    )

# VALIDATE TOTAL AVAILABLE POSTS

print("\n" + "=" * 80)
print("NEWER META COVERAGE VALIDATION")
print("=" * 80)

original_meta_posts = posts["total_views"].notna().sum()

daily_meta_posts = daily_meta["fb_total_views_data_available_posts"].sum()

print("Original posts with total_views:", original_meta_posts)
print("Aggregated available-data posts:", daily_meta_posts)

if original_meta_posts == daily_meta_posts:
    print("Meta feature coverage validation: PASSED")
else:
    raise ValueError("Mismatch in Meta feature coverage counts.")

# PREVIEW OUTPUT

print("\n" + "=" * 80)
print("PREVIEW OF FINAL FACEBOOK DAILY DATASET")
print("=" * 80)

print(final_daily.head())

print("\nFinal columns added in Step 3C:")

new_cols = [
    col for col in final_daily.columns
    if col not in daily.columns
]

for col in new_cols:
    print("-", col)

# FINAL DATASET VALIDATION

print("\n" + "=" * 80)
print("FINAL FACEBOOK DAILY DATASET VALIDATION")
print("=" * 80)

print("Final dataset shape:", final_daily.shape)

print("Date range:")
print("Start:", final_daily["date"].min())
print("End:", final_daily["date"].max())

print("Total Facebook daily rows:", len(final_daily))

print("Total Facebook posts represented:",
      final_daily["fb_posts_count"].sum())

# SAVE FINAL FACEBOOK DAILY FEATURES

final_daily.to_csv(output_file, index=False)

print("\n" + "=" * 80)
print("FINAL FACEBOOK DAILY FEATURES FILE SAVED")
print("=" * 80)

print("Saved as:", output_file)

print("\nStep 3C complete.")
print("Facebook daily feature engineering pipeline complete.")
print("\nNext stage:")
print("Merge Instagram + Facebook daily features.")