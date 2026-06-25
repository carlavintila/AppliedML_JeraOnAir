import pandas as pd
import numpy as np

# =========================================================
# STEP 3B:
# ADD POST TYPE COUNTS + CATEGORICAL FEATURES
# =========================================================

post_file = "facebook_posts_clean_all_years_sorted.csv"
step3a_file = "facebook_daily_features_step3A_core.csv"

output_file = "facebook_daily_features_step3B_categorical.csv"

# =========================================================
# LOAD FILES
# =========================================================

posts = pd.read_csv(post_file)
daily = pd.read_csv(step3a_file)

print("=" * 80)
print("STEP 3B: FACEBOOK CATEGORICAL DAILY FEATURES")
print("=" * 80)

print("\nPost-level shape:", posts.shape)
print("Step 3A daily shape:", daily.shape)

# =========================================================
# VALIDATE REQUIRED COLUMNS
# =========================================================

required_post_columns = [
    "publish_time",
    "permalink",
    "post_type",
    "is_crossposted",
    "avg_seconds_viewed",
    "source_file"
]

required_daily_columns = [
    "date",
    "fb_posts_count"
]

missing_post_cols = [col for col in required_post_columns if col not in posts.columns]
missing_daily_cols = [col for col in required_daily_columns if col not in daily.columns]

if missing_post_cols:
    raise ValueError("Missing columns in post-level file: " + str(missing_post_cols))

if missing_daily_cols:
    raise ValueError("Missing columns in Step 3A file: " + str(missing_daily_cols))

print("\nRequired column checks passed.")

# =========================================================
# CREATE DATE COLUMN FROM PUBLISH_TIME
# =========================================================

posts["publish_time_parsed"] = pd.to_datetime(
    posts["publish_time"],
    errors="coerce",
    utc=True
)

bad_dates = posts["publish_time_parsed"].isna().sum()

print("\nBad publish_time values:", bad_dates)

if bad_dates > 0:
    raise ValueError("Bad publish_time values found. Stop and inspect.")

posts["date"] = posts["publish_time_parsed"].dt.date.astype(str)
daily["date"] = pd.to_datetime(daily["date"], errors="coerce").dt.date.astype(str)

# =========================================================
# POST TYPE COUNTS
# =========================================================

print("\n" + "=" * 80)
print("POST TYPE INSPECTION")
print("=" * 80)

print(posts["post_type"].value_counts(dropna=False))

post_type_counts = (
    posts
    .pivot_table(
        index="date",
        columns="post_type",
        values="permalink",
        aggfunc="count",
        fill_value=0
    )
    .reset_index()
)

# Clean column names
rename_map = {}

for col in post_type_counts.columns:
    if col == "date":
        rename_map[col] = "date"
    else:
        clean_name = (
            str(col)
            .lower()
            .replace("fb-", "")
            .replace(" ", "_")
            .replace("-", "_")
        )
        rename_map[col] = f"fb_{clean_name}_posts_count"

post_type_counts = post_type_counts.rename(columns=rename_map)

print("\nPost type count columns created:")
for col in post_type_counts.columns:
    print("-", col)

# =========================================================
# CROSSPOSTED FEATURES
# =========================================================
# Important:
# Missing is_crossposted does not automatically mean 0.
# So we create both:
# - count of crossposted posts where data is available
# - number of posts where crosspost data is available

posts["is_crossposted_numeric"] = pd.to_numeric(
    posts["is_crossposted"],
    errors="coerce"
)

crossposted_daily = posts.groupby("date").agg(
    fb_crossposted_posts_count=("is_crossposted_numeric", "sum"),
    fb_crossposted_data_available_posts_count=("is_crossposted_numeric", "count")
).reset_index()

crossposted_daily["fb_crossposted_data_available"] = np.where(
    crossposted_daily["fb_crossposted_data_available_posts_count"] > 0,
    1,
    0
)

# =========================================================
# AVERAGE SECONDS VIEWED
# =========================================================
# This is already an average at post level.
# So daily aggregation should use mean, not sum.

posts["avg_seconds_viewed_numeric"] = pd.to_numeric(
    posts["avg_seconds_viewed"],
    errors="coerce"
)

avg_seconds_daily = posts.groupby("date").agg(
    fb_avg_seconds_viewed_mean=("avg_seconds_viewed_numeric", "mean")
).reset_index()

# =========================================================
# SOURCE FILE TRACEABILITY
# =========================================================

source_trace_daily = posts.groupby("date").agg(
    fb_source_files_count=("source_file", "nunique")
).reset_index()

# =========================================================
# MERGE ALL STEP 3B FEATURES INTO STEP 3A
# =========================================================

daily_3b = daily.copy()

daily_3b = daily_3b.merge(
    post_type_counts,
    on="date",
    how="left"
)

daily_3b = daily_3b.merge(
    crossposted_daily,
    on="date",
    how="left"
)

daily_3b = daily_3b.merge(
    avg_seconds_daily,
    on="date",
    how="left"
)

daily_3b = daily_3b.merge(
    source_trace_daily,
    on="date",
    how="left"
)

# =========================================================
# FILL SAFE COUNT FEATURES
# =========================================================

count_columns_to_fill = [
    col for col in daily_3b.columns
    if col.endswith("_posts_count")
]

for col in count_columns_to_fill:
    daily_3b[col] = daily_3b[col].fillna(0)

daily_3b["fb_source_files_count"] = daily_3b["fb_source_files_count"].fillna(0)

# Do NOT fill fb_avg_seconds_viewed_mean with 0.
# It is an average feature and should remain NaN if unavailable.

# =========================================================
# VALIDATION
# =========================================================

print("\n" + "=" * 80)
print("VALIDATION AFTER STEP 3B")
print("=" * 80)

print("Step 3A shape:", daily.shape)
print("Step 3B shape:", daily_3b.shape)

print("Duplicate dates:", daily_3b["date"].duplicated().sum())
print("Missing dates:", daily_3b["date"].isna().sum())

if daily_3b["date"].duplicated().sum() > 0:
    raise ValueError("Duplicate dates found after Step 3B merge.")

# Validate post type counts
post_type_count_cols = [
    col for col in daily_3b.columns
    if col.startswith("fb_") and col.endswith("_posts_count")
    and col not in [
        "fb_posts_count",
        "fb_shared_posts_count",
        "fb_promoted_reach_posts_count",
        "fb_crossposted_posts_count",
        "fb_crossposted_data_available_posts_count"
    ]
]

daily_3b["fb_post_type_total_check"] = daily_3b[post_type_count_cols].sum(axis=1)

post_type_total = daily_3b["fb_post_type_total_check"].sum()
original_total = len(posts)

print("\nOriginal post count:", original_total)
print("Total from post type counts:", post_type_total)

if original_total == post_type_total:
    print("Post type count validation: PASSED")
else:
    print("Post type count validation: FAILED")
    raise ValueError("Post type counts do not match original post count.")

# Validate Step 3A post count preserved
step3a_post_total = daily["fb_posts_count"].sum()
step3b_post_total = daily_3b["fb_posts_count"].sum()

print("\nStep 3A post total:", step3a_post_total)
print("Step 3B post total:", step3b_post_total)

if step3a_post_total == step3b_post_total:
    print("Step 3A post count preservation: PASSED")
else:
    raise ValueError("Step 3A post count changed after Step 3B merge.")

# Remove temporary validation column
daily_3b = daily_3b.drop(columns=["fb_post_type_total_check"])

# =========================================================
# PREVIEW
# =========================================================

print("\n" + "=" * 80)
print("PREVIEW OF STEP 3B OUTPUT")
print("=" * 80)

print(daily_3b.head())

print("\nNew Step 3B columns added:")
new_columns = [col for col in daily_3b.columns if col not in daily.columns]

for col in new_columns:
    print("-", col)

# =========================================================
# SAVE FILE
# =========================================================

daily_3b.to_csv(output_file, index=False)

print("\n" + "=" * 80)
print("STEP 3B FILE SAVED")
print("=" * 80)
print("Saved as:", output_file)

print("\nStep 3B complete.")
print("Next step will be Step 3C: add newer Meta-only features carefully with NaN preservation.")