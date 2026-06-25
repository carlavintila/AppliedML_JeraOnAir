import pandas as pd

# =========================================================
# INPUT FILES
# =========================================================
combined_2020_2024_file = "facebook_posts_clean_2020_2024.csv"
file_2019 = "facebook_posts_clean_2019.csv"

# =========================================================
# OUTPUT FILE
# =========================================================
output_file = "facebook_posts_clean_2019_2024.csv"

# =========================================================
# LOAD FILES
# =========================================================
df_2020_2024 = pd.read_csv(combined_2020_2024_file)
df_2019 = pd.read_csv(file_2019)

print("=" * 70)
print("Loaded files")
print("2020-2024 shape:", df_2020_2024.shape)
print("2019 shape:", df_2019.shape)

# =========================================================
# BASIC CHECKS BEFORE MERGING
# =========================================================
required_column = "publish_time"

if required_column not in df_2020_2024.columns:
    raise ValueError("publish_time is missing from 2020-2024 file.")

if required_column not in df_2019.columns:
    raise ValueError("publish_time is missing from 2019 file.")

print("\nMissing publish_time values:")
print("2020-2024:", df_2020_2024["publish_time"].isna().sum())
print("2019:", df_2019["publish_time"].isna().sum())

print("\nDuplicate full rows:")
print("2020-2024:", df_2020_2024.duplicated().sum())
print("2019:", df_2019.duplicated().sum())

if "permalink" in df_2020_2024.columns:
    print("\nDuplicate permalinks:")
    print("2020-2024:", df_2020_2024["permalink"].duplicated().sum())

if "permalink" in df_2019.columns:
    print("2019:", df_2019["permalink"].duplicated().sum())

# =========================================================
# ADD SOURCE FILE COLUMN TO 2019
# =========================================================
# The 2020-2024 file already has source_file.
# The 2019 file does not, so we add it for traceability.
df_2019["source_file"] = file_2019

# =========================================================
# ALIGN 2019 TO THE 2020-2024 STRUCTURE
# =========================================================
# We do NOT remove columns.
# We add missing columns to 2019 and then reorder columns.
for col in df_2020_2024.columns:
    if col not in df_2019.columns:
        df_2019[col] = pd.NA

# Reorder 2019 columns to match 2020-2024
df_2019 = df_2019[df_2020_2024.columns]

print("\nAfter alignment:")
print("2020-2024 columns:", len(df_2020_2024.columns))
print("2019 columns:", len(df_2019.columns))

if list(df_2019.columns) == list(df_2020_2024.columns):
    print("Column alignment check: OK")
else:
    raise ValueError("Column alignment failed.")

# =========================================================
# COMBINE FILES
# =========================================================
combined_df = pd.concat([df_2019, df_2020_2024], ignore_index=True)

print("\n" + "=" * 70)
print("MERGE COMPLETE")
print("Combined shape:", combined_df.shape)

# =========================================================
# VALIDATION AFTER MERGING
# =========================================================
expected_rows = len(df_2019) + len(df_2020_2024)
actual_rows = len(combined_df)

print("Expected rows:", expected_rows)
print("Actual rows:", actual_rows)

if expected_rows == actual_rows:
    print("Row count check: OK")
else:
    raise ValueError("Row count mismatch after merging.")

print("Duplicate full rows:", combined_df.duplicated().sum())

if "permalink" in combined_df.columns:
    print("Duplicate permalinks:", combined_df["permalink"].duplicated().sum())
    print("Missing permalinks:", combined_df["permalink"].isna().sum())

if "post_id" in combined_df.columns:
    print("Duplicate post_ids:", combined_df["post_id"].duplicated().sum())

# Check date parsing
parsed_dates = pd.to_datetime(combined_df["publish_time"], errors="coerce")
print("Unparseable publish_time values:", parsed_dates.isna().sum())

# Check source counts
if "source_file" in combined_df.columns:
    print("\nRows per source file:")
    print(combined_df["source_file"].value_counts())

# =========================================================
# SAVE OUTPUT
# =========================================================
combined_df.to_csv(output_file, index=False)

print("\nSaved combined file as:")
print(output_file)

print("\nPreview:")
print(combined_df.head())