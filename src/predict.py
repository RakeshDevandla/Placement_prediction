"""
Inference/Prediction Module for Placement Prediction
Use saved preprocessor and model for predictions on new data
"""

import os
import pickle
import pandas as pd
import numpy as np


class PlacementPredictor:
    """
    Make predictions using saved preprocessor and model
    Ensures no data leakage by using frozen transformations
    """
    
    def __init__(self, preprocessor_path, model_path):
        self.preprocessor = None
        self.model = None
        self.load_artifacts(preprocessor_path, model_path)
    
    def load_artifacts(self, preprocessor_path, model_path):
        """Load saved preprocessor and model"""
        # Load preprocessor
        with open(preprocessor_path, 'rb') as f:
            self.preprocessor = pickle.load(f)
        print(f"✓ Preprocessor loaded from {preprocessor_path}")
        
        # Load model
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        print(f"✓ Model loaded from {model_path}")
    
    def predict_single(self, data_dict):
        """
        Predict for a single candidate
        
        Args:
            data_dict: Dictionary with keys: Age, GPA, Experience, CGPA, Interview_Score, Technical_Skills
            
        Returns:
            prediction: 1 (Placed) or 0 (Not Placed)
            probability: Confidence score
        """
        # Convert to DataFrame
        df = pd.DataFrame([data_dict])
        
        # Apply preprocessing
        X_processed = self.preprocessor.transform(df)
        
        # Predict
        prediction = self.model.predict(X_processed)[0]
        
        # Get probability if available
        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(X_processed)[0]
            probability = proba[1]  # Probability of Placed
        else:
            probability = None
        
        return prediction, probability
    
    def predict_batch(self, df):
        """
        Predict for multiple candidates
        
        Args:
            df: DataFrame with required columns
            
        Returns:
            predictions: Array of predictions
            probabilities: Array of probabilities
        """
        # Drop ID column if present
        if 'ID' in df.columns:
            df = df.drop('ID', axis=1)
        
        # Apply preprocessing
        X_processed = self.preprocessor.transform(df)
        
        # Predict
        predictions = self.model.predict(X_processed)
        
        # Get probabilities
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(X_processed)[:, 1]
        else:
            probabilities = None
        
        return predictions, probabilities
    
    def predict_with_confidence(self, data_dict, confidence_threshold=0.7):
        """
        Predict with confidence levels
        """
        prediction, probability = self.predict_single(data_dict)
        
        status = "Placed" if prediction == 1 else "Not Placed"
        
        if probability is not None:
            confidence = "High" if probability >= confidence_threshold else "Low"
        else:
            confidence = "N/A"
        
        return {
            'prediction': status,
            'probability': probability,
            'confidence': confidence
        }


def example_single_prediction():
    """Example: Predict for a single candidate"""
    print("\n" + "="*80)
    print("EXAMPLE 1: SINGLE CANDIDATE PREDICTION")
    print("="*80)
    
    # Load artifacts
    predictor = PlacementPredictor(
        preprocessor_path='models/preprocessor.pkl',
        model_path='models/best_model.pkl'
    )
    
    # Example candidate
    candidate = {
        'Age': 25,
        'GPA': 3.5,
        'Experience': 2,
        'CGPA': 8.5,
        'Interview_Score': 75,
        'Technical_Skills': 'Python'
    }
    
    print(f"\nCandidate Profile:")
    for key, value in candidate.items():
        print(f"  {key}: {value}")
    
    result = predictor.predict_with_confidence(candidate)
    
    print(f"\nPrediction Result:")
    print(f"  Status: {result['prediction']}")
    if result['probability'] is not None:
        print(f"  Probability: {result['probability']:.2%}")
    print(f"  Confidence: {result['confidence']}")
    
    return result


def example_batch_prediction():
    """Example: Predict for multiple candidates"""
    print("\n" + "="*80)
    print("EXAMPLE 2: BATCH PREDICTION")
    print("="*80)
    
    # Load artifacts
    predictor = PlacementPredictor(
        preprocessor_path='models/preprocessor.pkl',
        model_path='models/best_model.pkl'
    )
    
    # Multiple candidates
    candidates_data = {
        'Age': [24, 26, 23, 27, 25],
        'GPA': [3.2, 3.8, 2.9, 3.5, 3.1],
        'Experience': [1, 3, 2, 4, 2],
        'CGPA': [7.8, 9.0, 7.5, 8.7, 8.2],
        'Interview_Score': [65, 85, 70, 90, 75],
        'Technical_Skills': ['Java', 'Python', 'C++', 'JavaScript', 'Java']
    }
    
    candidates_df = pd.DataFrame(candidates_data)
    
    print("\nCandidate Profiles:")
    print(candidates_df)
    
    predictions, probabilities = predictor.predict_batch(candidates_df)
    
    print("\nPrediction Results:")
    results_df = candidates_df.copy()
    results_df['Prediction'] = ['Placed' if p == 1 else 'Not Placed' for p in predictions]
    if probabilities is not None:
        results_df['Probability'] = [f"{p:.2%}" for p in probabilities]
    
    print(results_df)
    
    return results_df


if __name__ == "__main__":
    # Note: Run these examples from the project root directory
    # Change to src directory path if running from there
    
    print("PLACEMENT PREDICTION - INFERENCE MODULE")
    print("="*80)
    
    # Example 1: Single prediction
    # example_single_prediction()
    
    # Example 2: Batch prediction
    # example_batch_prediction()
    
    print("\nNote: To run predictions, ensure models are trained first via train_models.py")
