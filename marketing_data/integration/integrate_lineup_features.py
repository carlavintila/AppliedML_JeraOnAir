import pandas as pd

# =========================================================
# CREATE MODELING DATASET V1
# =========================================================

input_file = "master_calendar_with_temporal_features_2022_2026_FIXED.csv"
output_file = "modeling_dataset_v1.csv"

df = pd.read_csv(input_file)

print("=" * 80)
print("CREATE MODELING DATASET V1")
print("=" * 80)

print("\nOriginal shape:", df.shape)

# =========================================================
# COLUMNS TO REMOVE FOR FIRST BASELINE MODEL
# =========================================================

columns_to_remove = [
    # Zero-variance Instagram columns
    "ig_video_posts_count",
    "ig_other_posts_count",

    # Sparse Facebook Meta-only columns
    "fb_total_views_sum",
    "fb_organic_views_sum",
    "fb_promoted_views_sum",
    "fb_unique_impressions_sum",
    "fb_negative_feedback_sum",
    "fb_unique_negative_feedback_sum",
    "fb_has_promoted_views_posts_count",
    "marketing_total_views_meta_if_available",
]

existing_remove_cols = [col for col in columns_to_remove if col in df.columns]
missing_remove_cols = [col for col in columns_to_remove if col not in df.columns]

print("\nColumns selected for removal:")
for col in existing_remove_cols:
    print("-", col)

if missing_remove_cols:
    print("\nColumns not found, so they were skipped:")
    for col in missing_remove_cols:
        print("-", col)

# =========================================================
# CREATE MODELING DATASET
# =========================================================

model_df = df.drop(columns=existing_remove_cols)

print("\nNew shape:", model_df.shape)

# =========================================================
# VALIDATION
# =========================================================

print("\nDuplicate sale_date:", model_df["sale_date"].duplicated().sum())
print("Missing sale_date:", model_df["sale_date"].isna().sum())

if model_df["sale_date"].duplicated().sum() > 0:
    raise ValueError("Duplicate sale_date values found.")

if model_df["sale_date"].isna().sum() > 0:
    raise ValueError("Missing sale_date values found.")

remaining_nulls = model_df.isna().sum()
remaining_nulls = remaining_nulls[remaining_nulls > 0]

print("\nRemaining null values:")
if len(remaining_nulls) == 0:
    print("No null values remain.")
else:
    print(remaining_nulls)

# Check target
print("\nTarget check:")
print("tickets_sold column exists:", "tickets_sold" in model_df.columns)
print("Total tickets_sold:", model_df["tickets_sold"].sum())

# =========================================================
# SAVE FILE
# =========================================================

model_df.to_csv(output_file, index=False)

print("\n" + "=" * 80)
print("MODELING DATASET V1 SAVED")
print("=" * 80)
print("Saved as:", output_file)
print("\nModeling dataset v1 created successfully.")