# ============================================================
# LINEAR REGRESSION - PLACEMENT PREDICTION DATASET
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)


# ============================================================
# 1. FILE PATHS
# ============================================================

DATASET_PATH = (
    "/Users/kundan/Desktop/Machine Learning/"
    "placement_prediction/dataset/final_preprocess_M2.csv"
)

OUTPUT_FOLDER = (
    "/Users/kundan/Desktop/Machine Learning/"
    "placement_prediction/outputs/"
    "Linear_Regression_with_Metrics_M2"
)

IMAGE_FOLDER = os.path.join(
    OUTPUT_FOLDER,
    "images"
)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(IMAGE_FOLDER, exist_ok=True)


# ============================================================
# 2. LOAD DATASET
# ============================================================

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(
        f"Dataset not found: {DATASET_PATH}"
    )

df = pd.read_csv(DATASET_PATH)

print("=" * 60)
print("LINEAR REGRESSION - PLACEMENT PREDICTION")
print("=" * 60)

print("\nDataset Shape:")
print(df.shape)

print("\nFirst 5 Records:")
print(df.head())


# ============================================================
# 3. DISPLAY COLUMN NAMES
# ============================================================

print("\nDataset Columns:")

for column in df.columns:
    print(column)


# ============================================================
# 4. SELECT FEATURES AND TARGET
# ============================================================

# Multiple Linear Regression:
#
# x1 = CGPA
# x2 = AptitudeTestScore
# x3 = CodingTestScore
# x4 = MockInterviewScore

feature_columns = [
    "CGPA",
    "AptitudeTestScore",
    "CodingTestScore",
    "MockInterviewScore"
]

# Target variable
target_column = "PlacementStatus"


# ============================================================
# 5. CHECK REQUIRED COLUMNS
# ============================================================

required_columns = feature_columns + [target_column]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    print("\nERROR!")
    print("The following columns were not found:")
    print(missing_columns)

    print("\nAvailable columns are:")
    print(list(df.columns))

    raise ValueError(
        "Please change feature_columns and target_column "
        "according to your dataset."
    )


# ============================================================
# 6. CREATE MODEL DATA
# ============================================================

model_df = df[required_columns].copy()


# ============================================================
# 7. HANDLE INFINITE VALUES
# ============================================================

# Convert +inf and -inf into NaN so that the imputer
# can handle them.

model_df = model_df.replace(
    [np.inf, -np.inf],
    np.nan
)


# ============================================================
# 8. CHECK MISSING VALUES
# ============================================================

print("\n" + "=" * 60)
print("MISSING VALUE CHECK")
print("=" * 60)

print("\nMissing values in each column:")

print(
    model_df.isnull().sum()
)


# ============================================================
# 9. REMOVE ROWS WITH MISSING TARGET
# ============================================================

# The target value cannot be estimated using feature
# imputation, so rows with missing target values are removed.

target_missing = model_df[target_column].isnull().sum()

print(
    f"\nMissing target values: {target_missing}"
)

if target_missing > 0:

    model_df = model_df.dropna(
        subset=[target_column]
    )

    print(
        f"Removed {target_missing} rows "
        "with missing target values."
    )


# ============================================================
# 10. DEFINE X AND Y
# ============================================================

X = model_df[feature_columns]
y = model_df[target_column]


print("\nFeatures:")
print(feature_columns)

print("\nTarget:")
print(target_column)

print("\nTotal samples available:")
print(len(X))


# ============================================================
# 11. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\n" + "=" * 60)
print("TRAIN-TEST SPLIT")
print("=" * 60)

print(
    "\nTraining samples:",
    len(X_train)
)

print(
    "Testing samples:",
    len(X_test)
)


# ============================================================
# 12. HANDLE MISSING VALUES
# ============================================================

print("\n" + "=" * 60)
print("MISSING VALUE IMPUTATION")
print("=" * 60)

print("\nMissing values BEFORE imputation:")

print(
    X_train.isnull().sum()
)


# Median imputation
#
# IMPORTANT:
# The imputer is fitted ONLY on training data.
# This prevents test-data leakage.

imputer = SimpleImputer(
    strategy="median"
)


# Fit and transform training data

X_train = pd.DataFrame(
    imputer.fit_transform(X_train),
    columns=X_train.columns,
    index=X_train.index
)


# Transform test data using the same
# training-data statistics

X_test = pd.DataFrame(
    imputer.transform(X_test),
    columns=X_test.columns,
    index=X_test.index
)


print("\nMissing values AFTER imputation:")

print(
    X_train.isnull().sum()
)

print(
    X_test.isnull().sum()
)

print(
    "\nTotal missing values in X_train:",
    X_train.isnull().sum().sum()
)

print(
    "Total missing values in X_test:",
    X_test.isnull().sum().sum()
)


# ============================================================
# 13. CREATE LINEAR REGRESSION MODEL
# ============================================================

model = LinearRegression()


# ============================================================
# 14. TRAIN MODEL
# ============================================================

print("\n" + "=" * 60)
print("MODEL TRAINING")
print("=" * 60)

model.fit(
    X_train,
    y_train
)

print(
    "\nModel training completed successfully."
)


# ============================================================
# 15. MODEL COEFFICIENTS
# ============================================================

# Linear Regression equation:
#
# y = b0 + b1x1 + b2x2 + b3x3 + b4x4

print("\n" + "=" * 60)
print("MODEL COEFFICIENTS")
print("=" * 60)

print("\nIntercept (b0):")

print(
    model.intercept_
)


coefficient_df = pd.DataFrame({
    "Feature": feature_columns,
    "Coefficient": model.coef_
})


print("\nCoefficients:")

print(
    coefficient_df
)


# ============================================================
# 16. LINEAR REGRESSION EQUATION
# ============================================================

equation = (
    f"{target_column} = "
    f"{model.intercept_:.4f}"
)

for feature, coefficient in zip(
    feature_columns,
    model.coef_
):

    equation += (
        f" + ({coefficient:.4f} × {feature})"
    )


print("\nLinear Regression Equation:")

print(equation)


# ============================================================
# 17. PREDICTION
# ============================================================

y_pred = model.predict(
    X_test
)


# ============================================================
# 18. MODEL EVALUATION
# ============================================================

mae = mean_absolute_error(
    y_test,
    y_pred
)

mse = mean_squared_error(
    y_test,
    y_pred
)

rmse = np.sqrt(
    mse
)

r2 = r2_score(
    y_test,
    y_pred
)


print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

print(
    f"MAE  : {mae:.4f}"
)

print(
    f"MSE  : {mse:.4f}"
)

print(
    f"RMSE : {rmse:.4f}"
)

print(
    f"R²   : {r2:.4f}"
)


# ============================================================
# 19. CREATE PREDICTION RESULTS
# ============================================================

results = X_test.copy()

results["Actual"] = y_test

results["Predicted"] = y_pred

results["Residual"] = (
    results["Actual"] -
    results["Predicted"]
)

results["Absolute_Error"] = (
    results["Residual"].abs()
)


# ============================================================
# 20. SAVE PREDICTION RESULTS
# ============================================================

prediction_file = os.path.join(
    OUTPUT_FOLDER,
    "linear_regression_predictions.csv"
)

results.to_csv(
    prediction_file,
    index=False
)

print("\nPrediction results saved to:")

print(
    prediction_file
)


# ============================================================
# 21. SAVE MODEL COEFFICIENTS
# ============================================================

coefficient_file = os.path.join(
    OUTPUT_FOLDER,
    "linear_regression_coefficients.csv"
)

coefficient_df.to_csv(
    coefficient_file,
    index=False
)

print("\nCoefficient file saved to:")

print(
    coefficient_file
)


# ============================================================
# 22. SAVE MODEL METRICS
# ============================================================

metrics_df = pd.DataFrame({
    "Metric": [
        "MAE",
        "MSE",
        "RMSE",
        "R2"
    ],
    "Value": [
        mae,
        mse,
        rmse,
        r2
    ]
})


metrics_file = os.path.join(
    OUTPUT_FOLDER,
    "linear_regression_metrics.csv"
)

metrics_df.to_csv(
    metrics_file,
    index=False
)

print("\nMetrics file saved to:")

print(
    metrics_file
)


# ============================================================
# 23. ACTUAL VS PREDICTED GRAPH
# ============================================================

plt.figure(
    figsize=(8, 6)
)

plt.scatter(
    y_test,
    y_pred,
    alpha=0.6
)


# Perfect prediction line

minimum = min(
    y_test.min(),
    y_pred.min()
)

maximum = max(
    y_test.max(),
    y_pred.max()
)

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--"
)

plt.xlabel(
    "Actual Placement"
)

plt.ylabel(
    "Predicted Placement"
)

plt.title(
    "Linear Regression: Actual vs Predicted Placement"
)

plt.grid(
    True
)


actual_predicted_image = os.path.join(
    IMAGE_FOLDER,
    "actual_vs_predicted.png"
)

plt.savefig(
    actual_predicted_image,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 24. RESIDUAL GRAPH
# ============================================================

plt.figure(
    figsize=(8, 6)
)

plt.scatter(
    y_pred,
    results["Residual"],
    alpha=0.6
)

plt.axhline(
    y=0,
    linestyle="--"
)

plt.xlabel(
    "Predicted Placement"
)

plt.ylabel(
    "Residual"
)

plt.title(
    "Residual Plot - Linear Regression"
)

plt.grid(
    True
)


residual_image = os.path.join(
    IMAGE_FOLDER,
    "residual_plot.png"
)

plt.savefig(
    residual_image,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 25. COEFFICIENT GRAPH
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.bar(
    coefficient_df["Feature"],
    coefficient_df["Coefficient"]
)

plt.xlabel(
    "Features"
)

plt.ylabel(
    "Coefficient"
)

plt.title(
    "Linear Regression Feature Coefficients"
)

plt.xticks(
    rotation=30,
    ha="right"
)

plt.grid(
    axis="y"
)


coefficient_image = os.path.join(
    IMAGE_FOLDER,
    "feature_coefficients.png"
)

plt.savefig(
    coefficient_image,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 26. SAVE EQUATION
# ============================================================

equation_file = os.path.join(
    OUTPUT_FOLDER,
    "linear_regression_equation.txt"
)

with open(
    equation_file,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "Linear Regression Equation\n"
    )

    file.write(
        "=" * 40 + "\n"
    )

    file.write(
        equation + "\n"
    )


# ============================================================
# 27. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 60)
print("PROCESS COMPLETED SUCCESSFULLY")
print("=" * 60)

print("\nOutput Folder:")

print(
    OUTPUT_FOLDER
)

print("\nGenerated Files:")

print(
    "- linear_regression_predictions.csv"
)

print(
    "- linear_regression_coefficients.csv"
)

print(
    "- linear_regression_metrics.csv"
)

print(
    "- linear_regression_equation.txt"
)

print("\nGenerated Images:")

print(
    "- actual_vs_predicted.png"
)

print(
    "- residual_plot.png"
)

print(
    "- feature_coefficients.png"
)

print("\n" + "=" * 60)
print("ALL OPERATIONS COMPLETED")
print("=" * 60)