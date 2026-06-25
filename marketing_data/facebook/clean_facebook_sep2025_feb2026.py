import pandas as pd

# INPUT / OUTPUT
input_file = "Online Marketing Sep-01-2025_Feb-25-2026.xlsx"
output_file = "facebook_posts_clean_sep2025_feb2026.csv"

# LOAD RAW FILE
df = pd.read_excel(input_file)

print("Original shape:", df.shape)
print("Original columns:")
print(df.columns.tolist())

# STANDARDIZE DUPLICATE COLUMN NAMES SAFELY
def make_unique_columns(columns):
    seen = {}
    new_columns = []

    for col in columns:
        if col not in seen:
            seen[col] = 0
            new_columns.append(col)
        else:
            seen[col] += 1
            new_columns.append(f"{col}.{seen[col]}")

    return new_columns

df.columns = make_unique_columns(df.columns)

# COLUMN MAPPING FOR SEP 2025 - FEB 2026 FILE
column_mapping = {
    "Post ID": "post_id",
    "Page ID": "page_id",
    "Page Name": "page_name",
    "Page name": "page_name",

    "Title": "caption",
    "Description": "description",
    "Duration (sec.)": "duration_sec",
    "Duration (sec)": "duration_sec",

    "Publication time": "publish_time",
    "Publication Time": "publish_time",
    "Publish Time": "publish_time",

    "Permalink": "permalink",

    "Post Type": "post_type",
    "Post type": "post_type",

    "Is a Shared Post": "is_shared_post",
    "Is Shared Post": "is_shared_post",

    "Is Crossposted": "is_crossposted",

    "Caption Type": "caption_type",
    "Caption type": "caption_type",

    "Sponsored Content Status": "sponsored_content_status",

    # Views / impressions
    "Views": "total_views",
    "Organic Post Views": "organic_views",
    "Views of Promoted Posts": "promoted_views",
    "Organic Post Impressions": "organic_views",
    "Promoted Post Impressions": "promoted_views",

    # Reach
    "Range": "total_reach",
    "Reach": "total_reach",
    "IMPRESSION:UNIQUE_USERS": "unique_impressions",
    "Organic Post Reach": "organic_reach",
    "Promoted Post Reach": "promoted_reach",
    "Reach via Organic posts": "organic_reach",
    "Reach via Promoted Posts": "promoted_reach",

    # Engagement
    "Reactions, Comments and Shares": "total_engagement",
    "Reactions": "reactions",
    "Comments": "comments",
    "Shares": "shares",

    # Clicks
    "Total Clicks": "total_clicks",
    "Click on link": "link_clicks",
    "Link Clicks": "link_clicks",
    "Other Clicks": "other_clicks",
    "Consumption matching audience targeting (Video Click)": "video_clicks",
    "Consumption matching audience targeting (Photo Click)": "photo_clicks",
    "Consumption corresponding to audience targeting (Video Click)": "video_clicks",
    "Consumption corresponding to audience targeting (Photo Click)": "photo_clicks",

    # Video
    "3-Second Video Views": "video_views_3s",
    "3-second video views": "video_views_3s",
    "1-Minute Video Views": "video_views_1min",
    "1-minute video views": "video_views_1min",

    "Organic Post 3-Second Video Views": "organic_video_views_3s",
    "3-second video views of Organic posts": "organic_video_views_3s",

    "Promoted Post 3-Second Video Views": "promoted_video_views_3s",
    "3-second video views of Promoted posts": "promoted_video_views_3s",

    "Seconds Watched": "seconds_watched",
    "Seconds watched": "seconds_watched",

    "Average Seconds Watched": "avg_seconds_viewed",
    "Average Seconds viewed": "avg_seconds_viewed",

    # Ad / feedback
    "Estimated Earnings (USD)": "estimated_earnings_usd",
    "Estimated earnings (USD)": "estimated_earnings_usd",

    "Ad Impressions": "ad_impressions",
    "Ad views": "ad_impressions",

    "Ad CPM (USD)": "advertising_cpm_usd",
    "Advertising CPM (USD)": "advertising_cpm_usd",

    "Negative User Feedback": "negative_feedback",
    "Negative feedback from users": "negative_feedback",

    "Unique Negative User Feedback": "unique_negative_feedback",
    "Unique negative feedback from users": "unique_negative_feedback"
}

# keep only useful columns that exist
existing_cols = [col for col in column_mapping if col in df.columns]
df = df[existing_cols].copy()

# rename columns
df = df.rename(columns=column_mapping)

# STANDARD CLEAN SCHEMA
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
    "caption_type",
    "is_shared_post",
    "is_crossposted",
    "sponsored_content_status",

    "total_views",
    "organic_views",
    "promoted_views",

    "organic_reach",
    "promoted_reach",
    "total_reach",
    "unique_impressions",

    "total_engagement",
    "reactions",
    "comments",
    "shares",

    "total_clicks",
    "link_clicks",
    "other_clicks",
    "video_clicks",
    "photo_clicks",

    "video_views_3s",
    "video_views_1min",
    "organic_video_views_3s",
    "promoted_video_views_3s",

    "seconds_watched",
    "avg_seconds_viewed",

    "ad_impressions",
    "estimated_earnings_usd",
    "advertising_cpm_usd",
    "negative_feedback",
    "unique_negative_feedback",

    "has_promoted_reach",
    "has_promoted_views"
]

numeric_cols = [
    "duration_sec",

    "total_views",
    "organic_views",
    "promoted_views",

    "organic_reach",
    "promoted_reach",
    "total_reach",
    "unique_impressions",

    "total_engagement",
    "reactions",
    "comments",
    "shares",

    "total_clicks",
    "link_clicks",
    "other_clicks",
    "video_clicks",
    "photo_clicks",

    "video_views_3s",
    "video_views_1min",
    "organic_video_views_3s",
    "promoted_video_views_3s",

    "seconds_watched",
    "avg_seconds_viewed",

    "ad_impressions",
    "estimated_earnings_usd",
    "advertising_cpm_usd",
    "negative_feedback",
    "unique_negative_feedback",

    "has_promoted_reach",
    "has_promoted_views"
]

text_cols = [
    "post_id",
    "page_id",
    "page_name",
    "caption",
    "description",
    "permalink",
    "post_type",
    "caption_type",
    "is_shared_post",
    "is_crossposted",
    "sponsored_content_status"
]

# CREATE MISSING COLUMNS SAFELY
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

# CLEAN DATE
df["publish_time"] = pd.to_datetime(df["publish_time"], errors="coerce")

df["post_date"] = (
    df["publish_time"].dt.month.fillna(0).astype(int).astype(str) + "/" +
    df["publish_time"].dt.day.fillna(0).astype(int).astype(str) + "/" +
    df["publish_time"].dt.year.fillna(0).astype(int).astype(str)
)

df.loc[df["publish_time"].isna(), "post_date"] = ""

# CLEAN NUMERIC COLUMNS
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

# CREATE FLAGS
df["has_promoted_reach"] = (df["promoted_reach"] > 0).astype(int)
df["has_promoted_views"] = (df["promoted_views"] > 0).astype(int)

# CLEAN TEXT COLUMNS
for col in text_cols:
    df[col] = df[col].fillna("").astype(str).str.strip()

# STANDARDIZE FACEBOOK POST TYPES
def clean_facebook_post_type(x):
    x = str(x).lower().strip()

    if "reel" in x:
        return "FB-reel"
    elif "foto" in x or "photo" in x:
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

# FINAL ORDER
df = df[expected_cols]

# SAVE
df.to_csv(output_file, index=False)

# VALIDATION
print("\nSaved:", output_file)
print("Final shape:", df.shape)

print("\nDate range:")
print(df["publish_time"].min(), "to", df["publish_time"].max())

print("\nPost type distribution:")
print(df["post_type"].value_counts(dropna=False))

print("\nImportant totals:")
print("Total views:", df["total_views"].sum())
print("Organic views:", df["organic_views"].sum())
print("Promoted views:", df["promoted_views"].sum())
print("Total reach:", df["total_reach"].sum())
print("Organic reach:", df["organic_reach"].sum())
print("Promoted reach:", df["promoted_reach"].sum())
print("Total engagement:", df["total_engagement"].sum())
print("Reactions:", df["reactions"].sum())
print("Comments:", df["comments"].sum())
print("Shares:", df["shares"].sum())
print("Total clicks:", df["total_clicks"].sum())
print("Link clicks:", df["link_clicks"].sum())

print("\nDuplicate permalinks:", df["permalink"].duplicated().sum())
print("Empty permalinks:", (df["permalink"] == "").sum())

print("\nMissing values:")
print(df.isna().sum())