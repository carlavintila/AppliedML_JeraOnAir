import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from lightgbm import early_stopping
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score)
import matplotlib.pyplot as plt

# Step 3: Prediction
df = pd.read_csv("../datasets/final.csv")

df["sale_date"] = pd.to_datetime(df["sale_date"])

# choose to predict on log features or normal features
mode = "normal"

features = [
    "sales_lag_1",
    "sales_lag_7",
    "sales_lag_14",
    "sales_roll_3_prior",
    "sales_roll_7_prior",
    "sales_roll_14_prior",
    "sales_roll_30_prior",
    "cumulative_sales_prior",]

target = "tickets_sold"

train_df = df[df["festival_year"].isin([2022, 2023, 2024, 2025])].copy()
test_df = df[df["festival_year"] == 2026].copy()

X_train = train_df[features]
X_test = test_df[features]

# log-transform features

if mode == "log":
    train_df["target_log"] = np.log1p(train_df["tickets_sold"])
    test_df["target_log"] = np.log1p(test_df["tickets_sold"])
    y_train = train_df["target_log"]
    y_test = test_df["target_log"]
else:
    # for non-log features
    y_train = train_df[target]
    y_test = test_df[target]

# initialize model
model = LGBMRegressor(n_estimators=1000, learning_rate=0.02, max_depth=4, num_leaves=16, subsample=0.8, 
                       colsample_bytree=0.8, random_state=42, verbose = -1)

# training
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric="l1", callbacks=[early_stopping(100)])

# validation
if mode == "log":
    test_preds_log = model.predict(X_test)
    test_preds = np.expm1(test_preds_log)
    y_test = np.expm1(y_test)
else:
    test_preds = model.predict(X_test)

mae = mean_absolute_error(y_test, test_preds)
print("Test MAE:", mae)
rmse = np.sqrt(mean_squared_error(y_test,test_preds))
print("Test RMSE:", rmse)
r2 = r2_score(y_test, test_preds)
print("Test R2:", r2)

print(model.best_iteration_)

# feature importance
importance_df = pd.DataFrame({"feature": features, "importance": model.feature_importances_})
importance_df = importance_df.sort_values("importance", ascending=False)
print(importance_df)

# plots
plt.figure(figsize=(12,6))
plt.plot(test_df["sale_date"], y_test, label="actual")
plt.plot(test_df["sale_date"], test_preds, label="predicted")
plt.legend()
plt.show()