import pandas as pd
import os

files = [
    "../datasets/features_2022.csv",
    "../datasets/features_2023.csv",
    "../datasets/features_2024.csv",
    "../datasets/features_2025.csv",
    "../datasets/features_2026.csv",]

dfs = []

for file in files:
    # Extract year from filename (assumes format like features_2022.csv)
    year = os.path.basename(file).split("_")[1].split(".")[0]
    # Read CSV
    df = pd.read_csv(file)
    # Insert 'year' column at position 0
    df.insert(0, "year", int(year))
    dfs.append(df)

# Concatenate all DataFrames
combined_df = pd.concat(dfs, ignore_index=True)

output_path = "../datasets/combined_features.csv"
combined_df.to_csv(output_path, index=False)

print(f"Combined CSV saved to: {output_path}")