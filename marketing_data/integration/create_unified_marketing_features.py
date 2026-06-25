import pandas as pd
import numpy as np

# =========================================================
# STEP 3:
# CREATE UNIFIED ONLINE MARKETING FEATURES
# =========================================================

input_file = "master_calendar_with_instagram_facebook_2022_2026.csv"
output_file = "master_calendar_with_unified_marketing_2022_2026.csv"

df = pd.read_csv(input_file)

print("=" * 80)
print("STEP 3: UNIFIED ONLINE MARKETING FEATURE ENGINEERING")
print("=" * 80)

print("\nInput shape:", df.shape)

# =========================================================
# DATE CHECK
# =========================================================

df["sale_date"] = pd.to_datetime(df["sale_date"], errors="coerce")

if df["sale_date"].isna().sum() > 0:
    raise ValueError("Bad sale_date values found.")

if df["sale_date"].duplicated().sum() > 0:
    raise ValueError("Duplicate sale_date values found.")

print("Date validation: PASSED")

# =========================================================
# REQUIRED COLUMN CHECK
# =========================================================

required_cols = [
    "ig_posts_count",
    "ig_reach_sum",
    "ig_views_sum",
    "ig_likes_sum",
    "ig_comments_sum",
    "ig_shares_sum",
    "ig_saves_sum",
    "ig_has_post",

    "fb_posts_count",
    "fb_total_reach_sum",
    "fb_total_engagement_sum",
    "fb_reactions_sum",
    "fb_comments_sum",
    "fb_shares_sum",
    "fb_total_clicks_sum",
    "fb_link_clicks_sum",
    "fb_video_views_3s_sum",
    "fb_has_post"
]

missing_cols = [col for col in required_cols if col not in df.columns]

if missing_cols:
    raise ValueError("Missing required columns: " + str(missing_cols))

print("Required column check: PASSED")

# =========================================================
# MAKE SAFE NUMERIC COLUMNS
# =========================================================

for col in required_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

# =========================================================
# INSTAGRAM ENGAGEMENT
# =========================================================
# Instagram does not have one ready total engagement column.
# So we create it from available interaction metrics.

df["ig_total_engagement_sum"] = (
    df["ig_likes_sum"]
    + df["ig_comments_sum"]
    + df["ig_shares_sum"]
    + df["ig_saves_sum"]
)

# =========================================================
# UNIFIED CROSS-PLATFORM FEATURES
# =========================================================

# Total posting pressure
df["marketing_total_posts"] = (
    df["ig_posts_count"] + df["fb_posts_count"]
)

# Total reach pressure
df["marketing_total_reach"] = (
    df["ig_reach_sum"] + df["fb_total_reach_sum"]
)

# Total engagement pressure
df["marketing_total_engagement"] = (
    df["ig_total_engagement_sum"] + df["fb_total_engagement_sum"]
)

# Total visible/video exposure proxy
# Facebook's stable historical view feature is video_views_3s.
# Newer fb_total_views_sum is sparse, so we do NOT use it here.
df["marketing_total_views_core"] = (
    df["ig_views_sum"] + df["fb_video_views_3s_sum"]
)

# Total comments
df["marketing_total_comments"] = (
    df["ig_comments_sum"] + df["fb_comments_sum"]
)

# Total shares
df["marketing_total_shares"] = (
    df["ig_shares_sum"] + df["fb_shares_sum"]
)

# Facebook-only click intent signal
# Instagram file does not contain comparable click data.
df["marketing_total_clicks_known"] = df["fb_total_clicks_sum"]
df["marketing_link_clicks_known"] = df["fb_link_clicks_sum"]

# =========================================================
# ACTIVITY FLAGS
# =========================================================

df["marketing_activity_day"] = np.where(
    df["marketing_total_posts"] > 0,
    1,
    0
)

df["marketing_both_platforms_active"] = np.where(
    (df["ig_has_post"] > 0) & (df["fb_has_post"] > 0),
    1,
    0
)

df["marketing_only_instagram_active"] = np.where(
    (df["ig_has_post"] > 0) & (df["fb_has_post"] == 0),
    1,
    0
)

df["marketing_only_facebook_active"] = np.where(
    (df["ig_has_post"] == 0) & (df["fb_has_post"] > 0),
    1,
    0
)

df["marketing_platforms_active_count"] = (
    (df["ig_has_post"] > 0).astype(int)
    + (df["fb_has_post"] > 0).astype(int)
)

# =========================================================
# AVERAGE PER POST UNIFIED FEATURES
# =========================================================

df["marketing_avg_reach_per_post"] = np.where(
    df["marketing_total_posts"] > 0,
    df["marketing_total_reach"] / df["marketing_total_posts"],
    0
)

df["marketing_avg_engagement_per_post"] = np.where(
    df["marketing_total_posts"] > 0,
    df["marketing_total_engagement"] / df["marketing_total_posts"],
    0
)

df["marketing_avg_views_per_post_core"] = np.where(
    df["marketing_total_posts"] > 0,
    df["marketing_total_views_core"] / df["marketing_total_posts"],
    0
)

# =========================================================
# OPTIONAL META-AWARE FEATURE
# =========================================================
# This keeps Facebook Meta NaN logic.
# It should only be available when fb_total_views_sum is available.

if "fb_total_views_sum" in df.columns:
    df["marketing_total_views_meta_if_available"] = np.where(
        df["fb_total_views_sum"].notna(),
        df["ig_views_sum"] + df["fb_total_views_sum"],
        np.nan
    )

# =========================================================
# VALIDATION
# =========================================================

print("\n" + "=" * 80)
print("VALIDATION")
print("=" * 80)

print("Output shape:", df.shape)
print("Duplicate dates:", df["sale_date"].duplicated().sum())
print("Missing dates:", df["sale_date"].isna().sum())

print("\nMarketing activity days:", int(df["marketing_activity_day"].sum()))
print("No marketing activity days:", int(len(df) - df["marketing_activity_day"].sum()))

print("\nBoth platforms active days:", int(df["marketing_both_platforms_active"].sum()))
print("Only Instagram active days:", int(df["marketing_only_instagram_active"].sum()))
print("Only Facebook active days:", int(df["marketing_only_facebook_active"].sum()))

print("\nUnified feature totals:")
print("marketing_total_posts:", df["marketing_total_posts"].sum())
print("marketing_total_reach:", df["marketing_total_reach"].sum())
print("marketing_total_engagement:", df["marketing_total_engagement"].sum())
print("marketing_total_views_core:", df["marketing_total_views_core"].sum())

# Check impossible negatives
unified_cols = [
    "marketing_total_posts",
    "marketing_total_reach",
    "marketing_total_engagement",
    "marketing_total_views_core",
    "marketing_total_comments",
    "marketing_total_shares",
    "marketing_avg_reach_per_post",
    "marketing_avg_engagement_per_post",
    "marketing_avg_views_per_post_core"
]

negative_counts = {}

for col in unified_cols:
    negative_counts[col] = int((df[col] < 0).sum())

print("\nNegative value check:")
print(negative_counts)

if any(value > 0 for value in negative_counts.values()):
    raise ValueError("Negative values found in unified marketing features.")

# =========================================================
# FINAL NULL CHECK
# =========================================================

print("\n" + "=" * 80)
print("NULL CHECK")
print("=" * 80)

nulls = df.isna().sum()
nulls = nulls[nulls > 0]

if len(nulls) == 0:
    print("No null values.")
else:
    print(nulls)

# =========================================================
# SAVE FILE
# =========================================================

df["sale_date"] = df["sale_date"].dt.strftime("%Y-%m-%d")

df.to_csv(output_file, index=False)

print("\n" + "=" * 80)
print("FILE SAVED")
print("=" * 80)
print("Saved as:", output_file)

print("\nStep 3 complete.")
print("Next step: create lag and rolling features.")