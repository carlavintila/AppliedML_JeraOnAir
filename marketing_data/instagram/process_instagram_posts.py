import pandas as pd

# 1. File paths
file_1 = "Instagram Online Marketing Sep-01-2024_Sep-01-2025.csv"
file_2 = "Instagram Online Marketing Sep-01-2025_Feb-24-2026.csv"

# 2. Dutch to English column mapping
column_mapping = {
    "Bericht-ID": "post_id",
    "Gebruikersnaam account": "account_username",
    "Accountnaam": "account_name",
    "Omschrijving": "caption",
    "Duur (sec.)": "duration_sec",
    "Publicatietijdstip": "publish_time",
    "Permalink": "permalink",
    "Berichttype": "post_type",
    "Weergaven": "views",
    "Bereik": "reach",
    "Vind-ik-leuks": "likes",
    "Aantal keer gedeeld": "shares",
    "Opmerkingen": "comments",
    "Aantal keer opgeslagen": "saves",
    "Volgt": "follows"
}

# 3. Function to clean one Instagram file
def clean_instagram_file(file_path):
    print(f"\nReading file: {file_path}")

    # read csv
    df = pd.read_csv(file_path)

    print("Original columns:")
    print(df.columns.tolist())

    # keep only columns that exist in both the file and mapping
    available_columns = [col for col in column_mapping.keys() if col in df.columns]
    df = df[available_columns].copy()

    # rename columns
    df = df.rename(columns=column_mapping)

    # create any missing expected columns
    expected_columns = list(column_mapping.values())
    for col in expected_columns:
        if col not in df.columns:
            df[col] = pd.NA

    # reorder columns
    df = df[expected_columns]

    # 4. Convert publish_time and create post_date
    df["publish_time"] = pd.to_datetime(df["publish_time"], errors="coerce", utc=True)
    df["publish_time"] = df["publish_time"].dt.tz_convert("Europe/Amsterdam")
    df["post_date"] = df["publish_time"].dt.date

    # 5. Convert numeric columns
    numeric_columns = [
        "duration_sec", "views", "reach", "likes",
        "shares", "comments", "saves", "follows"
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # 6. Clean text columns
    text_columns = [
        "post_id", "account_username", "account_name",
        "caption", "permalink", "post_type"
    ]

    for col in text_columns:
        df[col] = df[col].astype("string").fillna("")

    # 7. Final column order
    df = df[
        [
            "post_id",
            "account_username",
            "account_name",
            "caption",
            "duration_sec",
            "publish_time",
            "post_date",
            "permalink",
            "post_type",
            "views",
            "reach",
            "likes",
            "shares",
            "comments",
            "saves",
            "follows"
        ]
    ]

    print("\nCleaned preview:")
    print(df.head())

    return df

# 8. Process both Instagram files
df_1 = clean_instagram_file(file_1)
df_2 = clean_instagram_file(file_2)

# combine
instagram_posts = pd.concat([df_1, df_2], ignore_index=True)

# 9. Remove exact duplicate posts if any
before_dedup = len(instagram_posts)
instagram_posts = instagram_posts.drop_duplicates(subset=["post_id"])
after_dedup = len(instagram_posts)

print(f"\nRows before deduplication: {before_dedup}")
print(f"Rows after deduplication: {after_dedup}")

# 10. Sort by publish time
instagram_posts = instagram_posts.sort_values("publish_time").reset_index(drop=True)

# 11. Save clean post-level dataset
output_file = "instagram_posts_clean.csv"
instagram_posts.to_csv(output_file, index=False)

print(f"\nDone. Clean Instagram post-level dataset saved as: {output_file}")

# 12. Basic checks
print("\nFinal columns:")
print(instagram_posts.columns.tolist())

print("\nShape:")
print(instagram_posts.shape)

print("\nMissing values per column:")
print(instagram_posts.isna().sum())

print("\nSample rows:")
print(instagram_posts.head(10))
