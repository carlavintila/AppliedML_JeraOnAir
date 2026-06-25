import pandas as pd
import os

# INPUT FILES
input_files = [
    "facebook_posts_clean_2020.csv",
    "facebook_posts_clean_2021.csv",
    "facebook_posts_clean_2022.csv",
    "facebook_posts_clean_2023.csv",
    "facebook_posts_clean_2024.csv",
]

# OUTPUT FILE
output_file = "facebook_posts_clean_2020_2024.csv"

# LOAD AND CHECK FILES
dataframes = []
base_columns = None

print("Checking Facebook files from 2020 to 2024...\n")

for file in input_files:
    df = pd.read_csv(file)

    print("=" * 70)
    print("File:", file)
    print("Shape:", df.shape)
    print("Number of columns:", len(df.columns))
    print("Duplicate full rows:", df.duplicated().sum())

    if "publish_time" in df.columns:
        print("Missing publish_time:", df["publish_time"].isna().sum())
    else:
        raise ValueError(f"publish_time column is missing in {file}")

    if base_columns is None:
        base_columns = list(df.columns)
        print("This file is used as the base structure.")
    else:
        current_columns = list(df.columns)

        if current_columns == base_columns:
            print("Column check: OK. Same columns and same order.")
        else:
            print("Column check: PROBLEM FOUND.")
            print("Columns are not the same as the base file.")

            missing_cols = set(base_columns) - set(current_columns)
            extra_cols = set(current_columns) - set(base_columns)

            print("Missing columns:", missing_cols)
            print("Extra columns:", extra_cols)

            raise ValueError("Column mismatch found. Stop merging.")

    # Add source file column.
    # This helps us know where each row came from later.
    df["source_file"] = file

    dataframes.append(df)

# COMBINE FILES
combined_df = pd.concat(dataframes, ignore_index=True)

print("\n" + "=" * 70)
print("MERGE COMPLETE")
print("Combined shape:", combined_df.shape)

# VALIDATION AFTER MERGING
expected_rows = sum(len(df) for df in dataframes)
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

if "post_id" in combined_df.columns:
    print("Duplicate post_ids:", combined_df["post_id"].duplicated().sum())

# Check whether publish_time can be read as dates
parsed_dates = pd.to_datetime(combined_df["publish_time"], errors="coerce")
print("Unparseable publish_time values:", parsed_dates.isna().sum())

# SAVE OUTPUT
combined_df.to_csv(output_file, index=False)

print("\nSaved combined file as:")
print(output_file)

print("\nPreview of combined data:")
print(combined_df.head())