import pandas as pd

# =========================================================
# INPUT / OUTPUT
# =========================================================
input_file = "FACEBOOK Jan-01-2021_Dec-31-2021_953854467041973.xlsx"
output_file = "facebook_posts_clean_2021.csv"

# =========================================================
# LOAD RAW FILE
# =========================================================
df = pd.read_excel(input_file)

print("Original shape:", df.shape)
print("Original columns:")
print(df.columns.tolist())

# =========================================================
# COLUMN MAPPING FOR 2021 FACEBOOK FILE
# =========================================================
column_mapping = {
    "Post ID": "post_id",
    "Page ID": "page_id",
    "Page Name": "page_name",
    "Title": "caption",
    "Description": "description",
    "Duration (sec.)": "duration_sec",
    "Publish Time": "publish_time",
    "Permalink": "permalink",
    "Post Type": "post_type",
    "Is Shared Post": "is_shared_post",
    "Is Crossposted": "is_crossposted",

    "Reactions, Comments and Shares": "total_engagement",
    "Reactions": "reactions",
    "Comments": "comments",
    "Shares": "shares",

    "Total Clicks": "total_clicks",
    "Link Clicks": "link_clicks",
    "Other Clicks": "other_clicks",
    "Consumption Matching Audience Targeting (Video Click)": "video_clicks",
    "Consumption Matching Audience Targeting (Photo Click)": "photo_clicks",

    "Reach": "total_reach",

    "Seconds Viewed": "seconds_watched",
    "Average Seconds Viewed": "avg_seconds_viewed",

    "Ad Impressions": "ad_impressions"
}

# keep only columns that exist
existing_cols = [col for col in column_mapping if col in df.columns]
df = df[existing_cols].copy()

# rename columns
df = df.rename(columns=column_mapping)

# =========================================================
# STANDARD FACEBOOK CLEAN SCHEMA
# =========================================================
expected_cols = [
    "post_id",
    "page_id",
    "page_name",
    "caption",
    "description",
    "duration_sec",
    "publish_time",
    "post_date",
    "permalink",
    "post_type",
    "is_shared_post",
    "is_crossposted",

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
    "avg_seconds_viewed",

    "ad_impressions",
    "has_promoted_reach"
]

numeric_cols = [
    "duration_sec",
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
    "avg_seconds_viewed",
    "ad_impressions",
    "has_promoted_reach"
]

text_cols = [
    "post_id",
    "page_id",
    "page_name",
    "caption",
    "description",
    "permalink",
    "post_type",
    "is_shared_post",
    "is_crossposted"
]

# =========================================================
# CREATE MISSING COLUMNS SAFELY
# =========================================================
for col in expected_cols:
    if col not in df.columns:
        if col in numeric_cols:
            df[col] = 0
        elif col == "publish_time":
            df[col] = pd.NaT
        elif col == "post_date":
            df[col] = ""
        else:
            df[col] = ""

# =========================================================
# CLEAN DATE
# =========================================================
df["publish_time"] = pd.to_datetime(df["publish_time"], errors="coerce")

df["post_date"] = (
    df["publish_time"].dt.month.fillna(0).astype(int).astype(str) + "/" +
    df["publish_time"].dt.day.fillna(0).astype(int).astype(str) + "/" +
    df["publish_time"].dt.year.fillna(0).astype(int).astype(str)
)

df.loc[df["publish_time"].isna(), "post_date"] = ""

# =========================================================
# CLEAN NUMERIC COLUMNS
# =========================================================
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

# 2021 only has total reach, not organic/promoted reach
df["has_promoted_reach"] = (df["promoted_reach"] > 0).astype(int)

# =========================================================
# CLEAN TEXT COLUMNS
# =========================================================
for col in text_cols:
    df[col] = df[col].fillna("").astype(str).str.strip()

# =========================================================
# STANDARDIZE FACEBOOK POST TYPES
# =========================================================
def clean_facebook_post_type(x):
    x = str(x).lower().strip()

    if "foto" in x or "photo" in x:
        return "FB-photo"
    elif "video" in x:
        return "FB-video"
    elif "link" in x:
        return "FB-link"
    elif "tekst" in x or "text" in x:
        return "FB-text"
    else:
        return "FB-other"

df["post_type"] = df["post_type"].apply(clean_facebook_post_type)

# =========================================================
# FINAL COLUMN ORDER
# =========================================================
df = df[expected_cols]

# =========================================================
# SAVE CLEAN FILE
# =========================================================
df.to_csv(output_file, index=False)

# =========================================================
# VALIDATION OUTPUT
# =========================================================
print("\nSaved:", output_file)
print("Final shape:", df.shape)

print("\nFinal columns:")
print(df.columns.tolist())

print("\nPost type distribution:")
print(df["post_type"].value_counts(dropna=False))

print("\nDate range:")
print(df["publish_time"].min(), "to", df["publish_time"].max())

print("\nImportant totals:")
print("Total engagement:", df["total_engagement"].sum())
print("Reactions:", df["reactions"].sum())
print("Comments:", df["comments"].sum())
print("Shares:", df["shares"].sum())
print("Total clicks:", df["total_clicks"].sum())
print("Link clicks:", df["link_clicks"].sum())
print("Total reach:", df["total_reach"].sum())
print("Ad impressions:", df["ad_impressions"].sum())

print("\nMissing values:")
print(df.isna().sum())