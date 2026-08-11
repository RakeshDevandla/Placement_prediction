"""
Model Training Suite for Placement Prediction
Implements multiple model families following strategic preprocessing matrix
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from preprocessing import PlacementPreprocessor


class ModelTrainer:
    """
    Train and evaluate multiple model families
    Follows strategic preprocessing selection matrix
    """
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
    
    def build_models(self):
        """
        Build all model families:
        1. Tree-Based: RandomForest, GradientBoosting
        2. Linear: LogisticRegression
        3. SVM: SVC (works well with scaling)
        """
        print("\n" + "="*80)
        print("BUILDING MODEL FAMILIES")
        print("="*80)
        
        self.models = {
            # Tree-Based Models (not sensitive to scaling, handles non-linearity)
            'RandomForest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            'GradientBoosting': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            ),
            # Linear Models (require scaling - handled in preprocessing)
            'LogisticRegression': LogisticRegression(
                max_iter=1000,
                random_state=42,
                solver='lbfgs'
            ),
            # SVM (requires scaling - handled in preprocessing)
            'SVM': SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                probability=True,
                random_state=42
            )
        }
        
        print("✓ Models configured:")
        for name in self.models.keys():
            print(f"  - {name}")
        
        return self.models
    
    def train_models(self, X_train, y_train):
        """Train all models"""
        print("\n" + "="*80)
        print("TRAINING MODELS")
        print("="*80)
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...", end=" ")
            model.fit(X_train, y_train)
            print("✓ Complete")
    
    def evaluate_models(self, X_test, y_test, X_train=None, y_train=None):
        """Evaluate all models on test set"""
        print("\n" + "="*80)
        print("MODEL EVALUATION")
        print("="*80)
        
        for name, model in self.models.items():
            # Predictions
            y_pred = model.predict(X_test)
            
            # Probabilities for ROC-AUC
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test)[:, 1]
            else:
                y_proba = model.decision_function(X_test)
            
            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_proba)
            
            # Store results
            self.results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'roc_auc': roc_auc,
                'y_pred': y_pred,
                'y_proba': y_proba,
                'confusion_matrix': confusion_matrix(y_test, y_pred)
            }
            
            print(f"\n{name}:")
            print(f"  Accuracy:  {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall:    {recall:.4f}")
            print(f"  F1-Score:  {f1:.4f}")
            print(f"  ROC-AUC:   {roc_auc:.4f}")
        
        # Find best model
        self.best_model_name = max(
            self.results.items(), 
            key=lambda x: x[1]['f1']
        )[0]
        self.best_model = self.models[self.best_model_name]
        
        print("\n" + "="*80)
        print(f"BEST MODEL: {self.best_model_name} (F1-Score: {self.results[self.best_model_name]['f1']:.4f})")
        print("="*80)
    
    def print_classification_report(self, y_test):
        """Print detailed classification report for best model"""
        print("\n" + "="*80)
        print(f"DETAILED CLASSIFICATION REPORT - {self.best_model_name}")
        print("="*80)
        y_pred = self.results[self.best_model_name]['y_pred']
        print(classification_report(y_test, y_pred, target_names=['Not Placed', 'Placed']))
    
    def plot_confusion_matrices(self, output_dir):
        """Plot confusion matrices for all models"""
        os.makedirs(output_dir, exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for idx, (name, result) in enumerate(self.results.items()):
            cm = result['confusion_matrix']
            sns.heatmap(
                cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                xticklabels=['Not Placed', 'Placed'],
                yticklabels=['Not Placed', 'Placed']
            )
            axes[idx].set_title(f'{name}\n(F1: {result["f1"]:.4f})')
            axes[idx].set_ylabel('True Label')
            axes[idx].set_xlabel('Predicted Label')
        
        plt.tight_layout()
        filepath = os.path.join(output_dir, 'confusion_matrices.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"✓ Confusion matrices saved to {filepath}")
        plt.close()
    
    def plot_roc_curves(self, output_dir, y_test):
        """Plot ROC curves for all models"""
        os.makedirs(output_dir, exist_ok=True)
        
        plt.figure(figsize=(10, 8))
        
        for name, result in self.results.items():
            y_proba = result['y_proba']
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            auc = result['roc_auc']
            plt.plot(fpr, tpr, label=f'{name} (AUC: {auc:.4f})', linewidth=2)
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=11)
        plt.grid(alpha=0.3)
        
        filepath = os.path.join(output_dir, 'roc_curves.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"✓ ROC curves saved to {filepath}")
        plt.close()
    
    def plot_metrics_comparison(self, output_dir):
        """Plot comparison of all metrics across models"""
        os.makedirs(output_dir, exist_ok=True)
        
        metrics_df = pd.DataFrame(self.results).T[['accuracy', 'precision', 'recall', 'f1', 'roc_auc']]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        metrics_df.plot(kind='bar', ax=ax, width=0.8)
        plt.xlabel('Model', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.title('Model Performance Comparison', fontsize=14, fontweight='bold')
        plt.legend(title='Metrics', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45)
        plt.ylim([0, 1])
        plt.grid(axis='y', alpha=0.3)
        
        filepath = os.path.join(output_dir, 'metrics_comparison.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"✓ Metrics comparison saved to {filepath}")
        plt.close()
    
    def save_best_model(self, filepath):
        """Save the best trained model"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.best_model, f)
        print(f"✓ Best model ({self.best_model_name}) saved to {filepath}")
    
    def save_all_models(self, directory):
        """Save all trained models"""
        os.makedirs(directory, exist_ok=True)
        for name, model in self.models.items():
            filepath = os.path.join(directory, f'{name.lower()}.pkl')
            with open(filepath, 'wb') as f:
                pickle.dump(model, f)
        print(f"✓ All models saved to {directory}")


def main():
    """Main training pipeline"""
    
    # Step 1: Preprocess data
    print("STEP 1: PREPROCESSING DATA")
    preprocessor = PlacementPreprocessor(test_size=0.2, random_state=42)
    
    dataset_path = os.path.join(
        os.path.dirname(__file__), 
        '..', 'dataset', 'placement_predict_50K_Raw.csv'
    )
    df = preprocessor.load_data(dataset_path)
    analysis_info = preprocessor.analyze_data(df)
    
    X, y = preprocessor.preprocess_features(df)
    X_train, X_test, y_train, y_test = preprocessor.fit_and_transform(X, y)
    
    # Save preprocessor
    preprocessor_path = os.path.join(
        os.path.dirname(__file__), 
        '..', 'models', 'preprocessor.pkl'
    )
    preprocessor.save_preprocessor(preprocessor_path)
    
    # Step 2: Train models
    print("\nSTEP 2: TRAINING MODELS")
    trainer = ModelTrainer()
    trainer.build_models()
    trainer.train_models(X_train, y_train)
    
    # Step 3: Evaluate models
    print("\nSTEP 3: EVALUATING MODELS")
    trainer.evaluate_models(X_test, y_test, X_train, y_train)
    trainer.print_classification_report(y_test)
    
    # Step 4: Visualizations
    print("\nSTEP 4: GENERATING VISUALIZATIONS")
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'models', 'visualizations')
    trainer.plot_confusion_matrices(output_dir)
    trainer.plot_roc_curves(output_dir, y_test)
    trainer.plot_metrics_comparison(output_dir)
    
    # Step 5: Save models
    print("\nSTEP 5: SAVING MODELS")
    best_model_path = os.path.join(
        os.path.dirname(__file__), 
        '..', 'models', 'best_model.pkl'
    )
    trainer.save_best_model(best_model_path)
    
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    trainer.save_all_models(models_dir)
    
    print("\n" + "="*80)
    print("TRAINING PIPELINE COMPLETE")
    print("="*80)
    
    return trainer, preprocessor, X_train, X_test, y_train, y_test


if __name__ == "__main__":
    trainer, preprocessor, X_train, X_test, y_train, y_test = main()
