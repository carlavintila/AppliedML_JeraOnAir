import pandas as pd
import numpy as np

# =========================================================
# STEP 3A:
# CREATE CORE FACEBOOK DAILY FEATURES
# =========================================================

# Input file
input_file = "facebook_posts_clean_all_years_sorted.csv"

# Output file for this stage only
output_file = "facebook_daily_features_step3A_core.csv"

# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv(input_file)

print("=" * 80)
print("STEP 3A: CORE FACEBOOK DAILY FEATURE ENGINEERING")
print("=" * 80)

print("\nOriginal Facebook post-level shape:")
print(df.shape)

# =========================================================
# VALIDATE REQUIRED COLUMNS
# =========================================================

required_columns = [
    "publish_time",
    "permalink",
    "total_engagement",
    "reactions",
    "comments",
    "shares",
    "total_clicks",
    "link_clicks",
    "other_clicks",
    "video_clicks",
    "photo_clicks",
    "organic_reach",
    "promoted_reach",
    "total_reach",
    "video_views_3s",
    "video_views_1min",
    "organic_video_views_3s",
    "promoted_video_views_3s",
    "seconds_watched",
    "duration_sec",
    "is_shared_post",
    "has_promoted_reach"
]

missing_required = []

for col in required_columns:
    if col not in df.columns:
        missing_required.append(col)

if len(missing_required) > 0:
    raise ValueError(
        "These required columns are missing from the file: "
        + str(missing_required)
    )

print("\nRequired column check passed.")

# =========================================================
# BASIC DATA INTEGRITY CHECKS
# =========================================================

print("\n" + "=" * 80)
print("POST-LEVEL VALIDATION BEFORE AGGREGATION")
print("=" * 80)

print("Rows:", len(df))
print("Missing publish_time:", df["publish_time"].isna().sum())
print("Duplicate full rows:", df.duplicated().sum())
print("Duplicate permalinks:", df["permalink"].duplicated().sum())

# Stop if duplicate permalinks exist
if df["permalink"].duplicated().sum() > 0:
    raise ValueError(
        "Duplicate permalinks found. Stop and inspect before daily aggregation."
    )

# =========================================================
# PARSE DATES SAFELY
# =========================================================

df["publish_time_parsed"] = pd.to_datetime(
    df["publish_time"],
    errors="coerce",
    utc=True
)

bad_dates = df["publish_time_parsed"].isna().sum()

print("Bad publish_time values:", bad_dates)

if bad_dates > 0:
    raise ValueError(
        "Some publish_time values could not be parsed. Stop and inspect dates."
    )

# Create daily date column
# Important:
# We are only extracting the day for Facebook aggregation.
# We are NOT standardizing all project date formats yet.
df["date"] = df["publish_time_parsed"].dt.date

print("Earliest Facebook date:", df["date"].min())
print("Latest Facebook date:", df["date"].max())

# =========================================================
# CORE NUMERIC COLUMNS
# =========================================================
# These columns have stable historical coverage.
# Missing values are safe to fill with 0 before daily summing.

core_sum_columns = [
    "total_engagement",
    "reactions",
    "comments",
    "shares",
    "total_clicks",
    "link_clicks",
    "other_clicks",
    "video_clicks",
    "photo_clicks",
    "organic_reach",
    "promoted_reach",
    "total_reach",
    "video_views_3s",
    "video_views_1min",
    "organic_video_views_3s",
    "promoted_video_views_3s",
    "seconds_watched",
    "duration_sec",
    "is_shared_post",
    "has_promoted_reach"
]

# Convert safely to numeric
for col in core_sum_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

# =========================================================
# DAILY CORE AGGREGATION
# =========================================================

daily_core = df.groupby("date").agg(

    # Basic post count
    fb_posts_count=("permalink", "count"),

    # Engagement
    fb_total_engagement_sum=("total_engagement", "sum"),
    fb_reactions_sum=("reactions", "sum"),
    fb_comments_sum=("comments", "sum"),
    fb_shares_sum=("shares", "sum"),

    # Clicks
    fb_total_clicks_sum=("total_clicks", "sum"),
    fb_link_clicks_sum=("link_clicks", "sum"),
    fb_other_clicks_sum=("other_clicks", "sum"),
    fb_video_clicks_sum=("video_clicks", "sum"),
    fb_photo_clicks_sum=("photo_clicks", "sum"),

    # Reach
    fb_organic_reach_sum=("organic_reach", "sum"),
    fb_promoted_reach_sum=("promoted_reach", "sum"),
    fb_total_reach_sum=("total_reach", "sum"),

    # Video metrics
    fb_video_views_3s_sum=("video_views_3s", "sum"),
    fb_video_views_1min_sum=("video_views_1min", "sum"),
    fb_organic_video_views_3s_sum=("organic_video_views_3s", "sum"),
    fb_promoted_video_views_3s_sum=("promoted_video_views_3s", "sum"),
    fb_seconds_watched_sum=("seconds_watched", "sum"),

    # Duration
    fb_duration_sec_sum=("duration_sec", "sum"),
    fb_duration_sec_mean=("duration_sec", "mean"),

    # Binary counts
    fb_shared_posts_count=("is_shared_post", "sum"),
    fb_promoted_reach_posts_count=("has_promoted_reach", "sum")

).reset_index()

# =========================================================
# AVERAGE PER POST FEATURES
# =========================================================

daily_core["fb_avg_engagement_per_post"] = (
    daily_core["fb_total_engagement_sum"] / daily_core["fb_posts_count"]
)

daily_core["fb_avg_reactions_per_post"] = (
    daily_core["fb_reactions_sum"] / daily_core["fb_posts_count"]
)

daily_core["fb_avg_comments_per_post"] = (
    daily_core["fb_comments_sum"] / daily_core["fb_posts_count"]
)

daily_core["fb_avg_shares_per_post"] = (
    daily_core["fb_shares_sum"] / daily_core["fb_posts_count"]
)

daily_core["fb_avg_clicks_per_post"] = (
    daily_core["fb_total_clicks_sum"] / daily_core["fb_posts_count"]
)

daily_core["fb_avg_reach_per_post"] = (
    daily_core["fb_total_reach_sum"] / daily_core["fb_posts_count"]
)

daily_core["fb_avg_seconds_watched_per_post"] = (
    daily_core["fb_seconds_watched_sum"] / daily_core["fb_posts_count"]
)

# Simple flag
daily_core["fb_has_post"] = 1

# =========================================================
# SORT DAILY DATA
# =========================================================

daily_core["date_sort"] = pd.to_datetime(daily_core["date"], errors="coerce")
daily_core = daily_core.sort_values("date_sort")
daily_core = daily_core.drop(columns=["date_sort"])

# =========================================================
# VALIDATION AFTER AGGREGATION
# =========================================================

print("\n" + "=" * 80)
print("VALIDATION AFTER STEP 3A DAILY AGGREGATION")
print("=" * 80)

print("Daily core shape:", daily_core.shape)

print("First daily date:", daily_core["date"].min())
print("Last daily date:", daily_core["date"].max())

print("Duplicate daily dates:", daily_core["date"].duplicated().sum())
print("Missing daily dates:", daily_core["date"].isna().sum())

original_post_count = len(df)
daily_post_count = daily_core["fb_posts_count"].sum()

print("\nOriginal post-level row count:", original_post_count)
print("Aggregated daily post count:", daily_post_count)

if original_post_count == daily_post_count:
    print("Post count validation: PASSED")
else:
    print("Post count validation: FAILED")
    raise ValueError("Post count mismatch after daily aggregation.")

# Validate important sums
validation_columns = {
    "total_engagement": "fb_total_engagement_sum",
    "reactions": "fb_reactions_sum",
    "comments": "fb_comments_sum",
    "shares": "fb_shares_sum",
    "total_clicks": "fb_total_clicks_sum",
    "link_clicks": "fb_link_clicks_sum",
    "total_reach": "fb_total_reach_sum",
    "video_views_3s": "fb_video_views_3s_sum",
    "seconds_watched": "fb_seconds_watched_sum"
}

print("\nCore numeric sum validation:")

for original_col, daily_col in validation_columns.items():

    original_sum = df[original_col].sum()
    daily_sum = daily_core[daily_col].sum()

    difference = original_sum - daily_sum

    print(
        original_col,
        "| original:",
        original_sum,
        "| daily:",
        daily_sum,
        "| difference:",
        difference
    )

    if abs(difference) > 0.0001:
        raise ValueError(
            f"Aggregation mismatch for {original_col}. "
            f"Difference: {difference}"
        )

print("\nCore numeric sum validation: PASSED")

# =========================================================
# PREVIEW OUTPUT
# =========================================================

print("\n" + "=" * 80)
print("PREVIEW OF STEP 3A OUTPUT")
print("=" * 80)

print(daily_core.head())

print("\nColumns created:")
for col in daily_core.columns:
    print("-", col)

# =========================================================
# SAVE FILE
# =========================================================

daily_core.to_csv(output_file, index=False)

print("\n" + "=" * 80)
print("STEP 3A FILE SAVED")
print("=" * 80)
print("Saved as:", output_file)

print("\nStep 3A complete.")
print("Next step will be Step 3B: add post type counts and safer categorical count features.")