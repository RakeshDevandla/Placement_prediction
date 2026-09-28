import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


# ============================================================
# 1. LOAD DATASET
# ============================================================

DATA_PATH = (
    "/Users/kundan/Desktop/Machine Learning/placement_prediction/"
    "dataset/final_preprocess_M2.csv"
)

data = pd.read_csv(DATA_PATH)

print("Dataset shape before cleaning:", data.shape)


# ============================================================
# 2. CHECK FOR MISSING / INVALID VALUES
# ============================================================

print("\nMissing values:")
print(data.isnull().sum())

print("\nTotal missing values:", data.isnull().sum().sum())


# Replace infinity values with NaN
data = data.replace([np.inf, -np.inf], np.nan)


# Remove rows containing NaN
before = len(data)

data = data.dropna()

after = len(data)

print("\nRows removed:", before - after)
print("Dataset shape after cleaning:", data.shape)


# ============================================================
# 3. EXTRACT FEATURES AND TARGET
# ============================================================

# All columns except last = features
X = data.iloc[:, :-1].values.astype(float)

# Last column = target
y = data.iloc[:, -1].values.astype(float)


print("\nFeature shape:", X.shape)
print("Target shape:", y.shape)

print("\nX contains NaN:", np.isnan(X).any())
print("y contains NaN:", np.isnan(y).any())

print("X contains Inf:", np.isinf(X).any())
print("y contains Inf:", np.isinf(y).any())


# ============================================================
# 4. CREATE IMAGE OUTPUT FOLDER
# ============================================================

IMAGE_FOLDER = (
    "/Users/kundan/Desktop/Machine Learning/placement_prediction/"
    "outputs/Linear_Regression_CFNE_GD_Compare_M2"
)

os.makedirs(IMAGE_FOLDER, exist_ok=True)

print("\nImage output folder:")
print(IMAGE_FOLDER)


# ============================================================
# 5. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# 6. FEATURE SCALING
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ============================================================
# 7. CLOSED FORM SOLUTION
#    NORMAL EQUATION
# ============================================================

# Add bias/intercept column

X_train_bias = np.c_[
    np.ones((X_train_scaled.shape[0], 1)),
    X_train_scaled
]

X_test_bias = np.c_[
    np.ones((X_test_scaled.shape[0], 1)),
    X_test_scaled
]


# ------------------------------------------------------------
# Normal Equation
#
# theta = (X^T X)^(-1) X^T y
#
# pinv() is used instead of inv() for numerical stability.
# ------------------------------------------------------------

theta_normal = np.linalg.pinv(
    X_train_bias.T @ X_train_bias
) @ X_train_bias.T @ y_train


# ============================================================
# 8. NORMAL EQUATION PREDICTION
# ============================================================

pred_normal = X_test_bias @ theta_normal


# Safety check

if not np.isfinite(pred_normal).all():
    raise ValueError(
        "Normal Equation produced NaN or infinite predictions."
    )


# ============================================================
# 9. NORMAL EQUATION METRICS
# ============================================================

mse_normal = mean_squared_error(
    y_test,
    pred_normal
)

r2_normal = r2_score(
    y_test,
    pred_normal
)


print("\n------ Closed Form Normal Equation ------")

print("\nCoefficients:")
print(theta_normal)

print("\nMSE:", mse_normal)
print("R2 Score:", r2_normal)


# ============================================================
# 10. GRADIENT DESCENT DATA
# ============================================================

X_train_gd = X_train_bias
X_test_gd = X_test_bias


m = len(y_train)


# Initialize parameters

theta_gd = np.zeros(
    X_train_gd.shape[1]
)


learning_rate = 0.01

epochs = 1000


# ============================================================
# 11. STORE LOSS
# ============================================================

loss_history = []


# ============================================================
# 12. GRADIENT DESCENT ITERATIONS
# ============================================================

for epoch in range(epochs):

    # Prediction
    predictions = X_train_gd @ theta_gd

    # Error
    errors = predictions - y_train

    # Gradient
    gradients = (
        (2 / m)
        * (X_train_gd.T @ errors)
    )

    # Update parameters
    theta_gd -= (
        learning_rate * gradients
    )

    # Calculate current MSE
    current_predictions = X_train_gd @ theta_gd

    loss = np.mean(
        (current_predictions - y_train) ** 2
    )

    loss_history.append(loss)


# ============================================================
# 13. GRADIENT DESCENT PREDICTION
# ============================================================

pred_gd = X_test_gd @ theta_gd


# Safety check

if not np.isfinite(pred_gd).all():
    raise ValueError(
        "Gradient Descent produced NaN or infinite predictions."
    )


# ============================================================
# 14. GRADIENT DESCENT METRICS
# ============================================================

mse_gd = mean_squared_error(
    y_test,
    pred_gd
)

r2_gd = r2_score(
    y_test,
    pred_gd
)


print("\n------ Gradient Descent ------")

print("\nCoefficients:")
print(theta_gd)

print("\nMSE:", mse_gd)
print("R2 Score:", r2_gd)


# ============================================================
# 15. COMPARISON
# ============================================================

print("\n==========================================")
print("           MODEL COMPARISON")
print("==========================================")

print("\nNormal Equation")
print("MSE =", mse_normal)
print("R2  =", r2_normal)

print("\nGradient Descent")
print("MSE =", mse_gd)
print("R2  =", r2_gd)


# ============================================================
# 16. IMAGE 1
# ACTUAL VS PREDICTED VALUES
# ============================================================

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    pred_normal,
    alpha=0.5,
    label="Normal Equation"
)

plt.scatter(
    y_test,
    pred_gd,
    alpha=0.5,
    label="Gradient Descent"
)


minimum = min(
    y_test.min(),
    pred_normal.min(),
    pred_gd.min()
)

maximum = max(
    y_test.max(),
    pred_normal.max(),
    pred_gd.max()
)


plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--",
    label="Perfect Prediction"
)


plt.xlabel("Actual Values")

plt.ylabel("Predicted Values")

plt.title(
    "Actual vs Predicted Values"
)

plt.legend()

plt.grid(True)

plt.tight_layout()


image1 = os.path.join(
    IMAGE_FOLDER,
    "actual_vs_predicted.png"
)

plt.savefig(
    image1,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nImage saved:")
print(image1)


# ============================================================
# 17. IMAGE 2
# RESIDUAL COMPARISON
# ============================================================

normal_residuals = (
    y_test - pred_normal
)

gd_residuals = (
    y_test - pred_gd
)


plt.figure(figsize=(9, 6))

plt.scatter(
    pred_normal,
    normal_residuals,
    alpha=0.5,
    label="Normal Equation"
)

plt.scatter(
    pred_gd,
    gd_residuals,
    alpha=0.5,
    label="Gradient Descent"
)


plt.axhline(
    y=0,
    linestyle="--"
)


plt.xlabel(
    "Predicted Values"
)

plt.ylabel(
    "Residuals"
)

plt.title(
    "Residual Comparison"
)

plt.legend()

plt.grid(True)

plt.tight_layout()


image2 = os.path.join(
    IMAGE_FOLDER,
    "residual_comparison.png"
)

plt.savefig(
    image2,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Image saved:")
print(image2)


# ============================================================
# 18. IMAGE 3
# GRADIENT DESCENT LOSS CURVE
# ============================================================

plt.figure(figsize=(9, 6))

plt.plot(
    range(1, epochs + 1),
    loss_history
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Mean Squared Error"
)

plt.title(
    "Gradient Descent Convergence"
)

plt.grid(True)

plt.tight_layout()


image3 = os.path.join(
    IMAGE_FOLDER,
    "gradient_descent_loss.png"
)

plt.savefig(
    image3,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Image saved:")
print(image3)


# ============================================================
# 19. SAVE IMAGE INFORMATION
# ============================================================

image_info = pd.DataFrame({

    "Image": [
        "actual_vs_predicted.png",
        "residual_comparison.png",
        "gradient_descent_loss.png"
    ],

    "Description": [
        "Actual values versus predictions from both methods",
        "Residual comparison between Normal Equation and Gradient Descent",
        "MSE loss across Gradient Descent epochs"
    ]
})


image_info.to_csv(
    os.path.join(
        IMAGE_FOLDER,
        "image_information.csv"
    ),
    index=False
)


# ============================================================
# 20. FINAL MESSAGE
# ============================================================

print("\n==========================================")
print("PROCESS COMPLETED SUCCESSFULLY")
print("==========================================")

print("\nAll images are stored in ONE folder:")

print(IMAGE_FOLDER)

print("\nGenerated images:")

print("1. actual_vs_predicted.png")
print("2. residual_comparison.png")
print("3. gradient_descent_loss.png")

print("\nOriginal dataset was NOT modified.")