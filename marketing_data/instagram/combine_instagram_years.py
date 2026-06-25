import pandas as pd

# =========================================================
# LOAD FILES
# =========================================================
df_old = pd.read_csv("instagram_posts_clean_all_years.csv", dtype=str)
df_new = pd.read_csv("instagram_posts_clean.csv", dtype=str)

# =========================================================
# PARSE publish_time SAFELY
# utc=True fixes mixed timezone issue
# =========================================================
df_old["publish_time"] = pd.to_datetime(df_old["publish_time"], errors="coerce", utc=True)
df_new["publish_time"] = pd.to_datetime(df_new["publish_time"], errors="coerce", utc=True)

# =========================================================
# KEEP ONLY 2025 AND 2026 FROM THE NEW FILE
# =========================================================
df_new = df_new[df_new["publish_time"].dt.year >= 2025].copy()

print("Old data shape:", df_old.shape)
print("New data after keeping only 2025+:", df_new.shape)

# =========================================================
# COMBINE BOTH FILES
# =========================================================
combined_df = pd.concat([df_old, df_new], ignore_index=True)

# =========================================================
# REMOVE DUPLICATES USING PERMALINK
# =========================================================
combined_df = combined_df.drop_duplicates(subset=["permalink"]).reset_index(drop=True)

# =========================================================
# SORT BY publish_time
# =========================================================
combined_df = combined_df.sort_values(by="publish_time").reset_index(drop=True)

# =========================================================
# FINAL VALIDATION
# =========================================================
print("\nFinal shape:", combined_df.shape)

print("\nDate range:")
print(combined_df["publish_time"].min(), "to", combined_df["publish_time"].max())

print("\nRows per year:")
print(combined_df["publish_time"].dt.year.value_counts().sort_index())

print("\nPost type distribution:")
print(combined_df["post_type"].value_counts(dropna=False))

print("\nMissing publish_time values:")
print(combined_df["publish_time"].isna().sum())

# =========================================================
# SAVE FINAL FILE
# =========================================================
combined_df.to_csv("instagram_posts_clean_full_2019_2026.csv", index=False)

print("\n✅ Final combined Instagram file saved as: instagram_posts_clean_full_2019_2026.csv")