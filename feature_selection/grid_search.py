import itertools
import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from lightgbm import early_stopping
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Step 3: Grid-search
df = pd.read_csv("../datasets/final.csv")
df["sale_date"] = pd.to_datetime(df["sale_date"])

# choose to predict on log features or normal features
mode = "normal"

target = "tickets_sold"

# Cross-validation setup:
# For each year in CV_YEARS, hold that year out for validation and train on all
# other CV_YEARS. Example: train on 2022-2024, validate on 2025; train on
# 2023-2025, validate on 2022; and so on.
CV_YEARS = [2022, 2023, 2024, 2025]
TEST_YEAR = 2026

# Always keep these features
BASE_FEATURES = [
    # time
    "days_since_sales_open",]

OPTIONAL_FEATURES = [
    "sales_lag_1",
    "sales_roll_3_prior",
    "sales_roll_7_prior",
    "sales_roll_14_prior",
    "is_announcement",
    "sales_lag_14",
    "Full Weekend_roll_3d",
    "marketing_total_engagement_roll_7_prior",
    "Full Weekend_lag_1d",
    "fb_total_engagement_sum_roll_7_prior",
    "fb_total_reach_sum_roll_7_prior",]

# Exhaustively grid-search subsets of the representative optional features up to this size.
# Set to None only if you really want every subset of OPTIONAL_FEATURES
MAX_ADDED_FEATURES = None

# Sanity checks
required_columns = ["festival_year", "sale_date", target]
missing_required_columns = [c for c in required_columns if c not in df.columns]
if missing_required_columns:
    raise ValueError(f"These required columns are missing from df.columns: {missing_required_columns}")
missing_features = [f for f in BASE_FEATURES + OPTIONAL_FEATURES if f not in df.columns]
if missing_features:
    raise ValueError(f"These features are missing from df.columns: {missing_features}")
available_cv_years = [year for year in CV_YEARS if year in set(df["festival_year"])]
missing_cv_years = sorted(set(CV_YEARS) - set(available_cv_years))
if missing_cv_years:
    print(f"Warning: these CV_YEARS are not present in the data and will be skipped: {missing_cv_years}")
if len(available_cv_years) < 2:
    raise ValueError("Need at least two available CV years to do held-out-year cross-validation.")
test_df = df[df["festival_year"] == TEST_YEAR].copy()
if test_df.empty:
    print(f"Warning: TEST_YEAR={TEST_YEAR} is not present in the data. Final test prediction is skipped.")


def make_model(n_estimators=1000):
    return LGBMRegressor(n_estimators=n_estimators, learning_rate=0.02, max_depth=4, num_leaves=16, subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1,)


def get_model_and_eval_targets(train_df, valid_df):
    """Return y arrays for training and validation.
    y_valid_model is the target scale used by LightGBM.
    y_valid_eval is always the original target scale used for MAE/RMSE/R2.
    """
    if mode == "log":
        y_train = np.log1p(train_df[target])
        y_valid_model = np.log1p(valid_df[target])
        y_valid_eval = valid_df[target]
    else:
        y_train = train_df[target]
        y_valid_model = valid_df[target]
        y_valid_eval = valid_df[target]

    return y_train, y_valid_model, y_valid_eval


def train_and_score_fold(features, validation_year):
    train_years = [year for year in available_cv_years if year != validation_year]
    train_df = df[df["festival_year"].isin(train_years)].copy()
    valid_df = df[df["festival_year"] == validation_year].copy()

    if train_df.empty:
        raise ValueError(f"No training rows found for validation_year={validation_year}.")
    if valid_df.empty:
        raise ValueError(f"No validation rows found for validation_year={validation_year}.")

    X_train = train_df[features]
    X_valid = valid_df[features]
    y_train, y_valid_model, y_valid_eval = get_model_and_eval_targets(train_df, valid_df)

    model = make_model()
    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid_model)], eval_metric="l1", callbacks=[early_stopping(100, verbose=False)],)

    valid_preds_model = model.predict(X_valid)
    if mode == "log":
        valid_preds = np.expm1(valid_preds_model)
    else:
        valid_preds = valid_preds_model

    mae = mean_absolute_error(y_valid_eval, valid_preds)
    rmse = np.sqrt(mean_squared_error(y_valid_eval, valid_preds))
    r2 = r2_score(y_valid_eval, valid_preds)

    pred_df = valid_df[["festival_year", "sale_date", target]].copy()
    pred_df["validation_year"] = validation_year
    pred_df["prediction"] = valid_preds
    pred_df["error"] = pred_df[target] - pred_df["prediction"]
    pred_df["abs_error"] = pred_df["error"].abs()

    return {
        "model": model,
        "train_years": train_years,
        "validation_year": validation_year,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "best_iteration": model.best_iteration_,
        "valid_preds_df": pred_df,}


def cross_validate_features(features):
    fold_scores = []
    fold_predictions = []
    models = []

    for validation_year in available_cv_years:
        fold = train_and_score_fold(features, validation_year)
        models.append(fold["model"])
        fold_predictions.append(fold["valid_preds_df"])
        fold_scores.append({
            "validation_year": validation_year,
            "train_years": ", ".join(map(str, fold["train_years"])),
            "mae": fold["mae"],
            "rmse": fold["rmse"],
            "r2": fold["r2"],
            "best_iteration": fold["best_iteration"],})
    fold_scores_df = pd.DataFrame(fold_scores)

    return {
        "models": models,
        "fold_scores": fold_scores_df,
        "fold_predictions": pd.concat(fold_predictions, ignore_index=True),
        "mean_mae": fold_scores_df["mae"].mean(),
        "std_mae": fold_scores_df["mae"].std(ddof=0),
        "mean_rmse": fold_scores_df["rmse"].mean(),
        "std_rmse": fold_scores_df["rmse"].std(ddof=0),
        "mean_r2": fold_scores_df["r2"].mean(),
        "std_r2": fold_scores_df["r2"].std(ddof=0),
        "mean_best_iteration": fold_scores_df["best_iteration"].mean(),}


def optional_feature_subsets(optional_features, max_added_features):
    if max_added_features is None:
        max_size = len(optional_features)
    else:
        max_size = min(max_added_features, len(optional_features))

    for k in range(0, max_size + 1):
        yield from itertools.combinations(optional_features, k)


results = []
fold_results = []
best = None

for added_features_tuple in optional_feature_subsets(OPTIONAL_FEATURES, MAX_ADDED_FEATURES):
    added_features = list(added_features_tuple)
    trial_features = BASE_FEATURES + added_features
    cv = cross_validate_features(trial_features)

    row = {
        "n_added_features": len(added_features),
        "added_features": ", ".join(added_features),
        "mean_mae": cv["mean_mae"],
        "std_mae": cv["std_mae"],
        "mean_rmse": cv["mean_rmse"],
        "std_rmse": cv["std_rmse"],
        "mean_r2": cv["mean_r2"],
        "std_r2": cv["std_r2"],
        "mean_best_iteration": cv["mean_best_iteration"],
        "n_total_features": len(trial_features),
    }
    results.append(row)

    fold_scores_for_trial = cv["fold_scores"].copy()
    fold_scores_for_trial["n_added_features"] = len(added_features)
    fold_scores_for_trial["added_features"] = ", ".join(added_features)
    fold_scores_for_trial["n_total_features"] = len(trial_features)
    fold_results.append(fold_scores_for_trial)

    if best is None or cv["mean_mae"] < best["mean_mae"]:
        best = {**row, "features": trial_features, "models": cv["models"], "fold_scores": cv["fold_scores"], "fold_predictions": cv["fold_predictions"],}
        print("New best:", row)

results_df = pd.DataFrame(results).sort_values("mean_mae", ascending=True)
results_df.to_csv("heldout_year_cv_representative_feature_grid_results.csv", index=False)

fold_results_df = pd.concat(fold_results, ignore_index=True)
fold_results_df.to_csv("heldout_year_cv_representative_feature_grid_fold_results.csv", index=False)

best["fold_scores"].to_csv("best_heldout_year_cv_fold_results.csv", index=False)
best["fold_predictions"].sort_values(["validation_year", "sale_date"]).to_csv("best_heldout_year_cv_predictions.csv",index=False,)

print("\nBest held-out-year CV result")
print("Validation years:", available_cv_years)
print("Mean validation MAE:", best["mean_mae"])
print("Std validation MAE:", best["std_mae"])
print("Mean validation RMSE:", best["mean_rmse"])
print("Std validation RMSE:", best["std_rmse"])
print("Mean validation R2:", best["mean_r2"])
print("Std validation R2:", best["std_r2"])
print("Mean best iteration:", best["mean_best_iteration"])
print("Added features:", best["added_features"] if best["added_features"] else "None; baseline only")
print("Total features:", len(best["features"]))

print("\nBest feature set fold results")
print(best["fold_scores"])

# Aggregate feature importance for the best feature set across the fold models.
importance_df = pd.DataFrame({"feature": best["features"]})
for i, model in enumerate(best["models"], start=1):
    importance_df[f"fold_{i}_importance"] = model.feature_importances_

importance_cols = [c for c in importance_df.columns if c.endswith("_importance")]
importance_df["mean_importance"] = importance_df[importance_cols].mean(axis=1)
importance_df["std_importance"] = importance_df[importance_cols].std(axis=1, ddof=0)
importance_df = importance_df.sort_values("mean_importance", ascending=False)
importance_df.to_csv("best_heldout_year_cv_feature_importance.csv", index=False)

print("\nAverage feature importance across CV folds")
print(importance_df)

# Optional final model fit on all CV years using the selected feature set.
# This is useful if you want one model for 2026 predictions after choosing features by CV.
# We use the rounded mean best iteration from CV so the final model has a similar amount of boosting.
final_train_df = df[df["festival_year"].isin(available_cv_years)].copy()
final_n_estimators = max(1, int(round(best["mean_best_iteration"])))
final_model = make_model(n_estimators=final_n_estimators)

if mode == "log":
    y_final_train = np.log1p(final_train_df[target])
else:
    y_final_train = final_train_df[target]

final_model.fit(final_train_df[best["features"]], y_final_train)

if not test_df.empty:
    test_preds_model = final_model.predict(test_df[best["features"]])
    if mode == "log":
        test_preds = np.expm1(test_preds_model)
    else:
        test_preds = test_preds_model

    test_predictions_df = test_df[["festival_year", "sale_date"]].copy()
    if target in test_df.columns:
        test_predictions_df[target] = test_df[target]
    test_predictions_df["prediction"] = test_preds
    test_predictions_df.sort_values("sale_date").to_csv("final_model_test_year_predictions.csv", index=False)

    print(f"\nSaved {TEST_YEAR} predictions from final model to final_model_test_year_predictions.csv")

# Plot held-out actual vs predicted for the best feature set.
plot_df = best["fold_predictions"].sort_values(["validation_year", "sale_date"])

for validation_year, year_df in plot_df.groupby("validation_year"):
    plt.figure(figsize=(12, 6))
    plt.plot(year_df["sale_date"], year_df[target], label="actual")
    plt.plot(year_df["sale_date"], year_df["prediction"], label="predicted")
    plt.title(f"Held-out validation year {validation_year}: actual vs predicted")
    plt.legend()
    plt.tight_layout()
    plt.show()