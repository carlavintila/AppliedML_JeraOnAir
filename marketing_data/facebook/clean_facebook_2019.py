import pandas as pd

input_file = "FACEBOOK Jan-01-2019_Dec-31-2019_1921309375414749.xlsx"
output_file = "facebook_posts_clean_2019.csv"

df = pd.read_excel(input_file)

print("Original shape:", df.shape)
print("Original columns:")
print(df.columns.tolist())

column_mapping = {
    "Message ID": "post_id",
    "Page ID": "page_id",
    "Page name": "page_name",
    "Title": "caption",
    "Description": "description",
    "Duration (sec.)": "duration_sec",
    "Publication time": "publish_time",
    "Permalink": "permalink",
    "Message type": "post_type",
    "Is a shared post": "is_shared_post",

    "Reactions, comments and shares": "total_engagement",
    "Comments": "reactions",
    "Comments.1": "comments",
    "Sharing actions": "shares",

    "Total number of clicks": "total_clicks",
    "Click on link": "link_clicks",
    "Other clicks": "other_clicks",
    "Consumption corresponding to audience targeting (Video Click)": "video_clicks",
    "Consumption corresponding to audience targeting (Photo Click)": "photo_clicks",

    "Reach via Organic posts": "organic_reach",
    "Reach via Promoted Posts": "promoted_reach",

    "3-second video views": "video_views_3s",
    "1-minute video views": "video_views_1min",
    "3-second video views of Organic posts": "organic_video_views_3s",
    "3-second video views of Promoted posts": "promoted_video_views_3s",

    "Seconds watched": "seconds_watched",
    "Average Seconds viewed": "avg_seconds_viewed"
}

existing_cols = [col for col in column_mapping if col in df.columns]
df = df[existing_cols].copy()
df = df.rename(columns=column_mapping)

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
    "is_shared_post"
]

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

df["publish_time"] = pd.to_datetime(df["publish_time"], errors="coerce")

df["post_date"] = (
    df["publish_time"].dt.month.fillna(0).astype(int).astype(str) + "/" +
    df["publish_time"].dt.day.fillna(0).astype(int).astype(str) + "/" +
    df["publish_time"].dt.year.fillna(0).astype(int).astype(str)
)

df.loc[df["publish_time"].isna(), "post_date"] = ""

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

df["total_reach"] = df["organic_reach"] + df["promoted_reach"]
df["has_promoted_reach"] = (df["promoted_reach"] > 0).astype(int)

for col in text_cols:
    df[col] = df[col].fillna("").astype(str).str.strip()

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

df = df[expected_cols]

df.to_csv(output_file, index=False)

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

print("\nMissing values:")
print(df.isna().sum())