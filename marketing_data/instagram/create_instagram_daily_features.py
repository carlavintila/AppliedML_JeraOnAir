import pandas as pd

# =========================================================
# LOAD FULL INSTAGRAM POST-LEVEL FILE
# =========================================================
input_file = "instagram_posts_clean_full_2019_2026.csv"
df = pd.read_csv(input_file)

print("Loaded file shape:", df.shape)
print("Columns:")
print(df.columns.tolist())

# =========================================================
# CLEAN DATA TYPES
# =========================================================
df["publish_time"] = pd.to_datetime(df["publish_time"], errors="coerce", utc=True)

# create sale_date in same style as your reference file: M/D/YYYY
df["sale_date"] = (
    df["publish_time"].dt.month.astype("Int64").astype(str) + "/" +
    df["publish_time"].dt.day.astype("Int64").astype(str) + "/" +
    df["publish_time"].dt.year.astype("Int64").astype(str)
)

# numeric columns
numeric_cols = ["views", "reach", "likes", "shares", "comments", "saves", "follows"]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

# =========================================================
# STANDARDIZE POST TYPES
# =========================================================
df["post_type"] = df["post_type"].fillna("").astype(str).str.strip()

# post type indicator columns
df["ig_reel_post"] = (df["post_type"] == "IG-reel").astype(int)
df["ig_image_post"] = (df["post_type"] == "Instagram-afbeelding").astype(int)
df["ig_carousel_post"] = (df["post_type"] == "Instagram-carrousel").astype(int)

# keep these columns to match your reference schema exactly
df["ig_video_post"] = (df["post_type"] == "IG-video").astype(int)
df["ig_other_post"] = (~df["post_type"].isin([
    "IG-reel",
    "Instagram-afbeelding",
    "Instagram-carrousel",
    "IG-video"
])).astype(int)

# each row is one post
df["ig_post"] = 1

# =========================================================
# AGGREGATE TO DAILY FEATURES
# =========================================================
daily_df = df.groupby("sale_date", as_index=False).agg({
    "ig_post": "sum",
    "views": "sum",
    "reach": "sum",
    "likes": "sum",
    "shares": "sum",
    "comments": "sum",
    "saves": "sum",
    "follows": "sum",
    "ig_reel_post": "sum",
    "ig_image_post": "sum",
    "ig_carousel_post": "sum",
    "ig_video_post": "sum",
    "ig_other_post": "sum"
})

# =========================================================
# RENAME COLUMNS TO MATCH YOUR REFERENCE FILE
# =========================================================
daily_df = daily_df.rename(columns={
    "ig_post": "ig_posts_count",
    "views": "ig_views_sum",
    "reach": "ig_reach_sum",
    "likes": "ig_likes_sum",
    "shares": "ig_shares_sum",
    "comments": "ig_comments_sum",
    "saves": "ig_saves_sum",
    "follows": "ig_follows_sum",
    "ig_reel_post": "ig_reel_posts_count",
    "ig_image_post": "ig_image_posts_count",
    "ig_carousel_post": "ig_carousel_posts_count",
    "ig_video_post": "ig_video_posts_count",
    "ig_other_post": "ig_other_posts_count"
})

# =========================================================
# CREATE AVERAGE FEATURES
# =========================================================
daily_df["ig_avg_views_per_post"] = daily_df["ig_views_sum"] / daily_df["ig_posts_count"]
daily_df["ig_avg_reach_per_post"] = daily_df["ig_reach_sum"] / daily_df["ig_posts_count"]
daily_df["ig_avg_likes_per_post"] = daily_df["ig_likes_sum"] / daily_df["ig_posts_count"]
daily_df["ig_avg_comments_per_post"] = daily_df["ig_comments_sum"] / daily_df["ig_posts_count"]
daily_df["ig_avg_shares_per_post"] = daily_df["ig_shares_sum"] / daily_df["ig_posts_count"]
daily_df["ig_avg_saves_per_post"] = daily_df["ig_saves_sum"] / daily_df["ig_posts_count"]

# post exists flag
daily_df["ig_has_post"] = (daily_df["ig_posts_count"] > 0).astype(int)

# =========================================================
# REORDER COLUMNS TO MATCH REFERENCE EXACTLY
# =========================================================
final_columns = [
    "sale_date",
    "ig_posts_count",
    "ig_views_sum",
    "ig_reach_sum",
    "ig_likes_sum",
    "ig_shares_sum",
    "ig_comments_sum",
    "ig_saves_sum",
    "ig_follows_sum",
    "ig_avg_views_per_post",
    "ig_avg_reach_per_post",
    "ig_avg_likes_per_post",
    "ig_avg_comments_per_post",
    "ig_avg_shares_per_post",
    "ig_avg_saves_per_post",
    "ig_reel_posts_count",
    "ig_image_posts_count",
    "ig_carousel_posts_count",
    "ig_video_posts_count",
    "ig_other_posts_count",
    "ig_has_post"
]

daily_df = daily_df[final_columns]

# =========================================================
# OPTIONAL: SORT BY REAL DATE
# =========================================================
daily_df["_sort_date"] = pd.to_datetime(daily_df["sale_date"], errors="coerce")
daily_df = daily_df.sort_values("_sort_date").drop(columns="_sort_date").reset_index(drop=True)

# =========================================================
# SAVE OUTPUT
# =========================================================
output_file = "instagram_daily_features_full_2019_2026.csv"
daily_df.to_csv(output_file, index=False)

# =========================================================
# VALIDATION
# =========================================================
print("\n✅ Daily features file created successfully!")
print("Output file:", output_file)
print("Shape:", daily_df.shape)
print("\nColumns:")
print(daily_df.columns.tolist())
print("\nFirst 5 rows:")
print(daily_df.head())
print("\nPost type totals:")
print("Reels:", daily_df["ig_reel_posts_count"].sum())
print("Images:", daily_df["ig_image_posts_count"].sum())
print("Carousels:", daily_df["ig_carousel_posts_count"].sum())
print("Videos:", daily_df["ig_video_posts_count"].sum())
print("Other:", daily_df["ig_other_posts_count"].sum())