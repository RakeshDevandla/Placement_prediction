# ---------------------------------------------------------
# Numeric Column Pre-processing Techniques
# Mean, Median, Mode
# Feature Scaling, Standardization, and Normalization
# Save all results in ONE CSV file
# ---------------------------------------------------------

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Step 1: Load Dataset
# ------------------------------------------------------------

file_path = '/Users/kundan/Desktop/Machine Learning/placement_prediction/dataset/placement_predict_50K_Raw.csv'

df = pd.read_csv(file_path)

print("Original Dataset")
print("----------------------------")
print(df.head())

print("Dataset Shape:", df.shape)

print("\nData Types:")
print("-------------------------------")
print(df.dtypes)

print("\nMissing Values:")
print("-------------------------------")
print(df.isnull().sum())

print("\nDuplicate Records:", df.duplicated().sum())


# ---------------------------------------------------
# Step 2: Remove Duplicate Records
# ---------------------------------------------------

df = df.drop_duplicates()


# ---------------------------------------------------
# Step 3: Handle Missing Values
# ---------------------------------------------------

# Numerical Columns
numerical_columns = df.select_dtypes(
    include=['int64', 'float64']
).columns

# Fill missing numerical values with Mean
for column in numerical_columns:
    df[column] = df[column].fillna(df[column].mean())


# Categorical Columns
categorical_columns = df.select_dtypes(
    include=['object']
).columns

# Fill missing categorical values with Mode
for column in categorical_columns:
    df[column] = df[column].fillna(df[column].mode()[0])


# ---------------------------------------------------
# Step 4: Remove Leading and Trailing Spaces
# ---------------------------------------------------

for column in categorical_columns:
    df[column] = df[column].str.strip()


# ------------------------------------------------------------
# Step 5: Select Numeric Columns
# ------------------------------------------------------------

numeric_columns = df.select_dtypes(
    include=['int64', 'float64']
).columns

print("\nNumeric Columns:")
print(list(numeric_columns))


# ------------------------------------------------------------
# Step 6: Standardization (Z-score)
# Mean = 0
# Standard Deviation = 1
# ------------------------------------------------------------

standard_scaler = StandardScaler()

standardized = standard_scaler.fit_transform(
    df[numeric_columns]
)

for i, col in enumerate(numeric_columns):
    df[col + "_Standardized"] = standardized[:, i]


# ------------------------------------------------------------
# Step 7: Feature Scaling (Min-Max Scaling)
# Values between 0 and 1
# ------------------------------------------------------------

minmax_scaler = MinMaxScaler()

scaled = minmax_scaler.fit_transform(
    df[numeric_columns]
)

for i, col in enumerate(numeric_columns):
    df[col + "_Scaled"] = scaled[:, i]


# ------------------------------------------------------------
# Step 8: Normalization (L2 Normalization)
# Each row becomes a unit vector
# ------------------------------------------------------------

normalizer = Normalizer(norm='l2')

normalized = normalizer.fit_transform(
    df[numeric_columns]
)

for i, col in enumerate(numeric_columns):
    df[col + "_Normalized"] = normalized[:, i]


# ------------------------------------------------------------
# Step 9: Display Results After Pre-processing
# ------------------------------------------------------------

print("\nDisplay Results after Preprocessed Dataset")
print("------------------------------------------")

print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Information:")
print(df.info())

print("\nColumns in Dataset:")
print(df.columns)

print("\nMissing Values After Preprocessing:")
print(df.isnull().sum())

print("\nDuplicate Records After Preprocessing:")
print(df.duplicated().sum())


# ---------------------------------------------------
# Step 10: Save Preprocessed Dataset
# ---------------------------------------------------

output_file = '/Users/kundan/Desktop/Machine Learning/placement_prediction/dataset/clean_minmax_stand_norma_M2.csv'

df.to_csv(output_file, index=False)

print("\nPreprocessed dataset saved successfully!")
print("File:", output_file)


# ---------------------------------------------------
# Step 11: Display Histogram
# ---------------------------------------------------

pf = pd.read_csv(output_file)

pf.hist(
    figsize=(12, 10),
    bins=10,
    edgecolor='black'
)

plt.suptitle(
    "Histogram of Preprocessed Placement Dataset"
)

plt.tight_layout()
plt.show()