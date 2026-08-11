"""
Flask Web Application for Placement Prediction
Integrates trained ML model with web interface
"""

import os
import sys
import pickle
import json
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from predict import PlacementPredictor

app = Flask(__name__)

# Global predictor instance
predictor = None


def load_predictor():
    """Load predictor on app startup"""
    global predictor
    
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    preprocessor_path = os.path.join(models_dir, 'preprocessor.pkl')
    model_path = os.path.join(models_dir, 'best_model.pkl')
    
    # Check if models exist
    if not os.path.exists(preprocessor_path) or not os.path.exists(model_path):
        print("WARNING: Models not found. Train models first using train_models.py")
        return False
    
    try:
        predictor = PlacementPredictor(preprocessor_path, model_path)
        print("✓ Models loaded successfully")
        return True
    except Exception as e:
        print(f"ERROR: Failed to load models - {e}")
        return False


@app.route('/')
def home():
    """Home page"""
    return render_template('home.html')


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    API endpoint for predictions
    Expects JSON with candidate data
    """
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['age', 'gpa', 'experience', 'cgpa', 'interview_score', 'technical_skills']
        if not all(field in data for field in required_fields):
            return jsonify({
                'error': 'Missing required fields',
                'required_fields': required_fields
            }), 400
        
        # Prepare candidate data
        candidate = {
            'Age': int(data['age']),
            'GPA': float(data['gpa']),
            'Experience': int(data['experience']),
            'CGPA': float(data['cgpa']),
            'Interview_Score': int(data['interview_score']),
            'Technical_Skills': data['technical_skills']
        }
        
        # Make prediction
        if predictor is None:
            return jsonify({
                'error': 'Predictor not initialized. Models may not be trained.'
            }), 500
        
        result = predictor.predict_with_confidence(candidate, confidence_threshold=0.7)
        
        return jsonify({
            'success': True,
            'prediction': result['prediction'],
            'probability': float(result['probability']) if result['probability'] else None,
            'confidence': result['confidence'],
            'candidate_info': candidate
        })
    
    except ValueError as e:
        return jsonify({'error': f'Invalid input format: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


@app.route('/api/batch_predict', methods=['POST'])
def batch_predict():
    """
    API endpoint for batch predictions
    Expects CSV data or JSON array
    """
    try:
        if 'file' in request.files:
            # Handle file upload
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
            
            # Read CSV
            df = pd.read_csv(file)
            
            # Drop ID column if present
            if 'ID' in df.columns:
                df = df.drop('ID', axis=1)
        else:
            # Handle JSON data
            data = request.get_json()
            if 'data' not in data:
                return jsonify({'error': 'Missing data field'}), 400
            
            df = pd.DataFrame(data['data'])
        
        # Make predictions
        if predictor is None:
            return jsonify({
                'error': 'Predictor not initialized. Models may not be trained.'
            }), 500
        
        predictions, probabilities = predictor.predict_batch(df)
        
        # Prepare response
        results = []
        for idx, (pred, prob) in enumerate(zip(predictions, probabilities)):
            results.append({
                'index': idx,
                'prediction': 'Placed' if pred == 1 else 'Not Placed',
                'probability': float(prob) if prob is not None else None
            })
        
        return jsonify({
            'success': True,
            'total_predictions': len(results),
            'results': results
        })
    
    except Exception as e:
        return jsonify({'error': f'Batch prediction failed: {str(e)}'}), 500


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': predictor is not None
    })


@app.route('/api/model_info', methods=['GET'])
def model_info():
    """Get information about loaded models"""
    if predictor is None:
        return jsonify({'error': 'Models not loaded'}), 500
    
    return jsonify({
        'model_loaded': True,
        'model_type': type(predictor.model).__name__,
        'technical_skills_options': ['Python', 'Java', 'C++', 'JavaScript'],
        'input_requirements': {
            'age': {'type': 'integer', 'range': [18, 65]},
            'gpa': {'type': 'float', 'range': [0, 4.0]},
            'experience': {'type': 'integer', 'range': [0, 50]},
            'cgpa': {'type': 'float', 'range': [0, 10.0]},
            'interview_score': {'type': 'integer', 'range': [0, 100]},
            'technical_skills': {'type': 'string', 'options': ['Python', 'Java', 'C++', 'JavaScript']}
        }
    })


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    print("="*80)
    print("PLACEMENT PREDICTION WEB APPLICATION")
    print("="*80)
    
    # Load predictor
    if load_predictor():
        print("\nStarting Flask application...")
        print("Open http://localhost:5000 in your browser")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("\nERROR: Cannot start application without trained models")
        print("Please run: python src/train_models.py")
        sys.exit(1)
