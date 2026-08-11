"""
Master Preprocessing Suite for Placement Prediction
Implements leak-proof pipeline following ML best practices
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle
import os


class PlacementPreprocessor:
    """
    Leak-proof preprocessing pipeline for placement prediction
    Follows scikit-learn best practices to avoid data leakage
    """
    
    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state
        self.preprocessor = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_data(self, filepath):
        """Load raw dataset"""
        print(f"Loading dataset from {filepath}")
        df = pd.read_csv(filepath)
        print(f"Dataset shape: {df.shape}")
        return df
    
    def analyze_data(self, df):
        """Analyze dataset for preprocessing requirements"""
        print("\n" + "="*80)
        print("DATA ANALYSIS")
        print("="*80)
        
        print(f"\nDataset Info:")
        print(df.info())
        
        print(f"\nMissing Values:")
        missing = df.isnull().sum()
        print(missing[missing > 0] if missing.sum() > 0 else "No missing values")
        
        print(f"\nNumerical Features Statistics:")
        print(df.describe())
        
        print(f"\nCategorical Features:")
        for col in df.select_dtypes(include='object').columns:
            if col not in ['Placement_Status']:
                print(f"  {col}: {df[col].nunique()} unique values")
                print(f"    Values: {df[col].unique()[:10]}")
        
        print(f"\nTarget Variable Distribution:")
        print(df['Placement_Status'].value_counts())
        
        return {
            'numeric_cols': df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_cols': df.select_dtypes(include='object').columns.tolist(),
            'missing_values': df.isnull().sum().sum()
        }
    
    def preprocess_features(self, df):
        """
        Separate features and target, handle ID column
        """
        # Drop ID column (not a feature)
        df = df.drop('ID', axis=1)
        
        # Separate features and target
        X = df.drop('Placement_Status', axis=1)
        y = df['Placement_Status']
        
        # Encode target: Placed=1, Not Placed=0
        y = (y == 'Placed').astype(int)
        
        return X, y
    
    def create_preprocessing_pipeline(self, X_sample):
        """
        Create scikit-learn ColumnTransformer pipeline
        This ensures transformations are fit only on training data
        """
        numeric_features = X_sample.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X_sample.select_dtypes(include='object').columns.tolist()
        
        # Numeric pipeline: Imputation → Standardization
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # Categorical pipeline: Imputation → One-Hot Encoding
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
        ])
        
        # Combine transformers
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ],
            remainder='passthrough'
        )
        
        self.preprocessor = preprocessor
        return preprocessor
    
    def fit_and_transform(self, X, y):
        """
        CRITICAL: Fit preprocessing on training data only
        This prevents data leakage
        """
        print("\n" + "="*80)
        print("TRAIN-TEST SPLIT & PREPROCESSING")
        print("="*80)
        
        # Step 1: Split BEFORE preprocessing (LEAK-PROOF approach)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.test_size, 
            random_state=self.random_state,
            stratify=y  # Maintain class distribution
        )
        
        print(f"\nTrain set size: {X_train.shape}")
        print(f"Test set size: {X_test.shape}")
        print(f"Train target distribution:\n{pd.Series(y_train).value_counts()}")
        print(f"Test target distribution:\n{pd.Series(y_test).value_counts()}")
        
        # Step 2: Create pipeline
        self.create_preprocessing_pipeline(X_train)
        
        # Step 3: Fit ONLY on training data
        X_train_transformed = self.preprocessor.fit_transform(X_train)
        
        # Step 4: Apply fitted transformations to test data
        X_test_transformed = self.preprocessor.transform(X_test)
        
        print(f"\nTransformed training set shape: {X_train_transformed.shape}")
        print(f"Transformed test set shape: {X_test_transformed.shape}")
        
        self.X_train = X_train_transformed
        self.X_test = X_test_transformed
        self.y_train = y_train.values if hasattr(y_train, 'values') else y_train
        self.y_test = y_test.values if hasattr(y_test, 'values') else y_test
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def save_preprocessor(self, filepath):
        """Save fitted preprocessor for inference"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.preprocessor, f)
        print(f"Preprocessor saved to {filepath}")
    
    def load_preprocessor(self, filepath):
        """Load saved preprocessor"""
        with open(filepath, 'rb') as f:
            self.preprocessor = pickle.load(f)
        print(f"Preprocessor loaded from {filepath}")
    
    def transform_new_data(self, X_new):
        """Apply preprocessing to new data (using fitted preprocessor)"""
        if self.preprocessor is None:
            raise ValueError("Preprocessor not fitted. Call fit_and_transform first.")
        return self.preprocessor.transform(X_new)


def main():
    """Main execution"""
    # Initialize preprocessor
    preprocessor = PlacementPreprocessor(test_size=0.2, random_state=42)
    
    # Load data
    dataset_path = os.path.join(
        os.path.dirname(__file__), 
        '..', 'dataset', 'placement_predict_50K_Raw.csv'
    )
    df = preprocessor.load_data(dataset_path)
    
    # Analyze data
    analysis_info = preprocessor.analyze_data(df)
    
    # Preprocess features and target
    X, y = preprocessor.preprocess_features(df)
    
    # Create and apply preprocessing pipeline
    X_train, X_test, y_train, y_test = preprocessor.fit_and_transform(X, y)
    
    # Save preprocessor
    preprocessor_path = os.path.join(
        os.path.dirname(__file__), 
        '..', 'models', 'preprocessor.pkl'
    )
    preprocessor.save_preprocessor(preprocessor_path)
    
    print("\n" + "="*80)
    print("PREPROCESSING COMPLETE")
    print("="*80)
    print(f"✓ Data loaded and analyzed")
    print(f"✓ Train-test split (stratified): {X_train.shape[0]} train, {X_test.shape[0]} test")
    print(f"✓ Leak-proof preprocessing applied")
    print(f"✓ Preprocessor saved for inference")
    
    return preprocessor, X_train, X_test, y_train, y_test


if __name__ == "__main__":
    preprocessor, X_train, X_test, y_train, y_test = main()
