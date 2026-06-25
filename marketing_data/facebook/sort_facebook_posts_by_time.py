import pandas as pd

# =========================================================
# INPUT / OUTPUT
# =========================================================
input_file = "facebook_posts_clean_all_years_clean_order.csv"
output_file = "facebook_posts_clean_all_years_sorted.csv"

# =========================================================
# LOAD FILE
# =========================================================
df = pd.read_csv(input_file)

print("=" * 70)
print("FILE LOADED")
print("=" * 70)
print("Original shape:", df.shape)

# =========================================================
# BASIC VALIDATION BEFORE SORTING
# =========================================================
if "publish_time" not in df.columns:
    raise ValueError("publish_time column is missing.")

if "permalink" not in df.columns:
    raise ValueError("permalink column is missing.")

print("\nMissing publish_time:", df["publish_time"].isna().sum())
print("Missing permalinks:", df["permalink"].isna().sum())
print("Duplicate permalinks before sorting:", df["permalink"].duplicated().sum())

# =========================================================
# PARSE PUBLISH TIME
# =========================================================
# utc=True safely handles timezone information.
df["publish_time_parsed"] = pd.to_datetime(
    df["publish_time"],
    errors="coerce",
    utc=True
)

bad_dates = df["publish_time_parsed"].isna().sum()
print("Unparseable publish_time values:", bad_dates)

if bad_dates > 0:
    raise ValueError("Some publish_time values could not be parsed. Stop and inspect them first.")

# =========================================================
# SORT BY DATE AND TIME
# =========================================================
df_sorted = df.sort_values(
    by="publish_time_parsed",
    ascending=True
).reset_index(drop=True)

# =========================================================
# OPTIONAL: UPDATE post_date SAFELY
# =========================================================
# This creates a clean date column in YYYY-MM-DD format.
# It is useful later when creating daily features.
df_sorted["post_date"] = df_sorted["publish_time_parsed"].dt.strftime("%Y-%m-%d")

# =========================================================
# REMOVE TEMPORARY SORTING COLUMN
# =========================================================
df_sorted = df_sorted.drop(columns=["publish_time_parsed"])

# =========================================================
# VALIDATION AFTER SORTING
# =========================================================
print("\n" + "=" * 70)
print("VALIDATION AFTER SORTING")
print("=" * 70)

print("Sorted shape:", df_sorted.shape)
print("Rows unchanged:", len(df_sorted) == len(df))

print("Duplicate permalinks after sorting:", df_sorted["permalink"].duplicated().sum())

check_dates = pd.to_datetime(
    df_sorted["publish_time"],
    errors="coerce",
    utc=True
)

print("Is sorted by publish_time:", check_dates.is_monotonic_increasing)

print("\nFirst date:", check_dates.min())
print("Last date:", check_dates.max())

if "source_file" in df_sorted.columns:
    print("\nRows per source file:")
    print(df_sorted["source_file"].value_counts())

# =========================================================
# SAVE FILE
# =========================================================
df_sorted.to_csv(output_file, index=False)

print("\n" + "=" * 70)
print("SORTED FILE SAVED")
print("=" * 70)
print("Saved as:", output_file)

print("\nPreview:")
print(df_sorted.head())