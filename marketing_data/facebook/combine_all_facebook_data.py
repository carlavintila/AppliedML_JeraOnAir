import pandas as pd

# =========================================================
# INPUT FILES
# =========================================================
file_2019_2024 = "facebook_posts_clean_2019_2024.csv"
file_sep2024_sep2025 = "facebook_posts_clean_sep2024_sep2025.csv"
file_sep2025_feb2026 = "facebook_posts_clean_sep2025_feb2026.csv"

# =========================================================
# OUTPUT FILE
# =========================================================
output_file = "facebook_posts_clean_all_years_clean_order.csv"

# =========================================================
# LOAD FILES
# =========================================================
df_2019_2024 = pd.read_csv(file_2019_2024)
df_sep2024_sep2025 = pd.read_csv(file_sep2024_sep2025)
df_sep2025_feb2026 = pd.read_csv(file_sep2025_feb2026)

print("=" * 70)
print("FILES LOADED")
print("=" * 70)
print("2019-2024 shape:", df_2019_2024.shape)
print("Sep2024-Sep2025 shape:", df_sep2024_sep2025.shape)
print("Sep2025-Feb2026 shape:", df_sep2025_feb2026.shape)

# =========================================================
# ADD SOURCE FILE COLUMN IF MISSING
# =========================================================
if "source_file" not in df_2019_2024.columns:
    df_2019_2024["source_file"] = file_2019_2024

if "source_file" not in df_sep2024_sep2025.columns:
    df_sep2024_sep2025["source_file"] = file_sep2024_sep2025

if "source_file" not in df_sep2025_feb2026.columns:
    df_sep2025_feb2026["source_file"] = file_sep2025_feb2026

# =========================================================
# CREATE CLEAN COLUMN ORDER
# =========================================================
# First keep the original 2019-2024 column order
base_columns = list(df_2019_2024.columns)

# Then add only new columns from the later files at the end
extra_columns = []

for df in [df_sep2024_sep2025, df_sep2025_feb2026]:
    for col in df.columns:
        if col not in base_columns and col not in extra_columns:
            extra_columns.append(col)

final_columns = base_columns + extra_columns

print("\nBase columns:", len(base_columns))
print("Extra new columns:", len(extra_columns))
print("Final columns:", len(final_columns))

print("\nExtra columns added at the end:")
print(extra_columns)

# =========================================================
# ALIGN ALL FILES TO SAME STRUCTURE
# =========================================================
def align_columns(df, final_cols):
    df = df.copy()

    for col in final_cols:
        if col not in df.columns:
            df[col] = pd.NA

    return df[final_cols]

df_2019_2024 = align_columns(df_2019_2024, final_columns)
df_sep2024_sep2025 = align_columns(df_sep2024_sep2025, final_columns)
df_sep2025_feb2026 = align_columns(df_sep2025_feb2026, final_columns)

print("\nColumn alignment complete.")
print("2019-2024 aligned shape:", df_2019_2024.shape)
print("Sep2024-Sep2025 aligned shape:", df_sep2024_sep2025.shape)
print("Sep2025-Feb2026 aligned shape:", df_sep2025_feb2026.shape)

# =========================================================
# COMBINE FILES
# =========================================================
combined_df = pd.concat(
    [df_2019_2024, df_sep2024_sep2025, df_sep2025_feb2026],
    ignore_index=True
)

print("\n" + "=" * 70)
print("MERGE COMPLETE")
print("=" * 70)
print("Combined shape BEFORE deduplication:", combined_df.shape)

# =========================================================
# VALIDATION BEFORE DEDUPLICATION
# =========================================================
print("\nDuplicate full rows:", combined_df.duplicated().sum())

if "permalink" not in combined_df.columns:
    raise ValueError("permalink column is missing. Cannot safely deduplicate.")

print("Missing permalinks:", combined_df["permalink"].isna().sum())
print("Duplicate permalinks:", combined_df["permalink"].duplicated().sum())

# =========================================================
# REMOVE DUPLICATES USING PERMALINK
# =========================================================
before_dedup = len(combined_df)

combined_df = combined_df.drop_duplicates(
    subset="permalink",
    keep="first"
)

after_dedup = len(combined_df)

print("\nDuplicates removed using permalink:", before_dedup - after_dedup)
print("Combined shape AFTER deduplication:", combined_df.shape)

# =========================================================
# DATE VALIDATION
# =========================================================
parsed_dates = pd.to_datetime(
    combined_df["publish_time"],
    errors="coerce"
)

print("\nUnparseable publish_time values:", parsed_dates.isna().sum())

# =========================================================
# SOURCE FILE COUNTS
# =========================================================
print("\nRows per source file:")
print(combined_df["source_file"].value_counts())

# =========================================================
# FINAL VALIDATION
# =========================================================
print("\nFinal duplicate full rows:", combined_df.duplicated().sum())
print("Final duplicate permalinks:", combined_df["permalink"].duplicated().sum())
print("Final columns:", len(combined_df.columns))

# =========================================================
# SAVE FINAL FILE
# =========================================================
combined_df.to_csv(output_file, index=False)

print("\n" + "=" * 70)
print("FINAL FILE SAVED")
print("=" * 70)
print("Saved as:", output_file)

print("\nPreview:")
print(combined_df.head())