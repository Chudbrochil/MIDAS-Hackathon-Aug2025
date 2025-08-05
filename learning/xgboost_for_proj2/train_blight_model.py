"""
Blight Detection Model Training Script
=====================================

This script trains XGBoost models for multi-class blight detection using
synthetic parcel data. Provides comprehensive evaluation metrics and 
model performance analysis for both small and large datasets.

Usage:
    python train_blight_model.py --small-data blight_data_small.csv --large-data blight_data_large.csv
    python train_blight_model.py --data blight_data_small.csv  # Train on single dataset
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import argparse
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support, roc_auc_score, log_loss,
    cohen_kappa_score, balanced_accuracy_score
)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

class BlightModelTrainer:
    """
    XGBoost model trainer for blight detection with comprehensive evaluation.
    """
    
    def __init__(self, random_state=RANDOM_STATE):
        self.random_state = random_state
        self.model = None
        self.feature_names = None
        self.scaler = None
        
    def load_data(self, filepath):
        """Load and validate dataset."""
        print(f"📂 Loading data from {filepath}")
        
        if not Path(filepath).exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")
            
        df = pd.read_csv(filepath)
        print(f"   Loaded {len(df):,} samples with {len(df.columns)} columns")
        
        # Validate required columns
        required_cols = ['parcel_id', 'blight_level']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        # Validate blight levels
        valid_levels = set([1, 2, 3, 4])
        actual_levels = set(df['blight_level'].unique())
        if not actual_levels.issubset(valid_levels):
            raise ValueError(f"Invalid blight levels found: {actual_levels - valid_levels}")
            
        print(f"   Blight distribution: {dict(df['blight_level'].value_counts().sort_index())}")
        return df
    
    def prepare_features(self, df):
        """Prepare features for model training."""
        print("🔧 Preparing features...")
        
        # Separate features and target
        feature_cols = [col for col in df.columns if col not in ['parcel_id', 'blight_level']]
        X = df[feature_cols].copy()
        y = df['blight_level'].copy()
        
        # Check for missing values
        missing_counts = X.isnull().sum()
        if missing_counts.sum() > 0:
            print(f"   ⚠️  Found missing values:")
            for col, count in missing_counts[missing_counts > 0].items():
                print(f"      {col}: {count} missing")
            # Fill missing values with median for numeric columns
            X = X.fillna(X.median())
        
        # Store feature names
        self.feature_names = feature_cols
        
        print(f"   Features prepared: {len(feature_cols)} columns")
        print(f"   Target classes: {sorted(y.unique())}")
        
        return X, y
    
    def train_model(self, X_train, y_train, X_val=None, y_val=None):
        """Train XGBoost model with optimal parameters."""
        print("🎯 Training XGBoost model...")
        
        # XGBoost parameters optimized for multi-class classification
        params = {
            'objective': 'multi:softprob',
            'num_class': 4,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 1,
            'random_state': self.random_state,
            'eval_metric': ['mlogloss', 'merror'],
            'verbosity': 0
        }
        
        # Create DMatrix for XGBoost (convert to 0-based indexing)
        dtrain = xgb.DMatrix(X_train, label=y_train-1)
        
        # Setup validation if provided
        evallist = [(dtrain, 'train')]
        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val-1)
            evallist.append((dval, 'val'))
        
        # Train model with early stopping
        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=1000,
            evals=evallist,
            early_stopping_rounds=50,
            verbose_eval=False
        )
        
        print(f"   Model trained with {self.model.num_boosted_rounds()} rounds")
        return self.model
    
    def predict(self, X):
        """Generate predictions."""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_model() first.")
            
        dtest = xgb.DMatrix(X)
        y_pred_proba = self.model.predict(dtest)
        y_pred = np.argmax(y_pred_proba, axis=1) + 1  # Convert back to 1-based indexing
        
        return y_pred, y_pred_proba
    
    def evaluate_model(self, X_test, y_test, dataset_name="Dataset"):
        """Perform comprehensive model evaluation."""
        print(f"\n{'='*60}")
        print(f"COMPREHENSIVE EVALUATION - {dataset_name.upper()}")
        print(f"{'='*60}")
        
        # Generate predictions
        y_pred, y_pred_proba = self.predict(X_test)
        
        # Basic metrics
        accuracy = accuracy_score(y_test, y_pred)
        balanced_acc = balanced_accuracy_score(y_test, y_pred)
        
        print(f"\n📊 BASIC METRICS:")
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   Balanced Accuracy: {balanced_acc:.4f}")
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, average=None)
        
        print(f"\n📋 PER-CLASS METRICS:")
        blight_labels = ['No Blight', 'Minor Blight', 'Moderate Blight', 'Severe Blight']
        for i, (class_label, label_name) in enumerate(zip([1, 2, 3, 4], blight_labels)):
            if i < len(precision):  # Check if class exists in predictions
                print(f"   Class {class_label} ({label_name}) - n={support[i]:,}:")
                print(f"     Precision: {precision[i]:.4f}")
                print(f"     Recall: {recall[i]:.4f}")
                print(f"     F1-Score: {f1[i]:.4f}")
        
        # Macro and weighted averages
        prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')
        prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        
        print(f"\n📈 AVERAGED METRICS:")
        print(f"   Macro Avg    - Precision: {prec_macro:.4f}, Recall: {rec_macro:.4f}, F1: {f1_macro:.4f}")
        print(f"   Weighted Avg - Precision: {prec_weighted:.4f}, Recall: {rec_weighted:.4f}, F1: {f1_weighted:.4f}")
        
        # Multi-class AUC metrics
        try:
            y_test_bin = label_binarize(y_test, classes=[1, 2, 3, 4])
            if y_test_bin.shape[1] > 1:  # Multi-class case
                auc_ovr = roc_auc_score(y_test_bin, y_pred_proba, multi_class='ovr', average='macro')
                auc_ovo = roc_auc_score(y_test_bin, y_pred_proba, multi_class='ovo', average='macro')
                print(f"\n🎯 MULTI-CLASS AUC:")
                print(f"   AUC (One-vs-Rest): {auc_ovr:.4f}")
                print(f"   AUC (One-vs-One): {auc_ovo:.4f}")
            else:
                print(f"\n🎯 MULTI-CLASS AUC:")
                print(f"   AUC calculation skipped (insufficient classes in test set)")
        except Exception as e:
            print(f"\n🎯 MULTI-CLASS AUC:")
            print(f"   AUC calculation failed: {str(e)[:50]}...")
        
        # Additional metrics
        try:
            logloss = log_loss(y_test, y_pred_proba, labels=[1, 2, 3, 4])
            print(f"   Log Loss: {logloss:.4f}")
        except:
            print(f"   Log Loss: calculation failed")
            
        kappa = cohen_kappa_score(y_test, y_pred)
        print(f"   Cohen's Kappa: {kappa:.4f}")
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred, labels=[1, 2, 3, 4])
        print(f"\n🔍 CONFUSION MATRIX:")
        print("     Predicted")
        print("     1    2    3    4")
        for i, row in enumerate(cm):
            print(f"  {i+1} {str(row).replace('[', ' ').replace(']', '')}")
        
        # Feature Importance
        if hasattr(self.model, 'get_score'):
            importance = self.model.get_score(importance_type='weight')
            if importance:
                sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
                
                print(f"\n🔧 TOP 10 FEATURE IMPORTANCE:")
                for i, (feature, score) in enumerate(sorted_importance[:10]):
                    # Map feature index to name
                    if feature.startswith('f') and feature[1:].isdigit():
                        feature_idx = int(feature[1:])
                        if feature_idx < len(self.feature_names):
                            feature_name = self.feature_names[feature_idx]
                        else:
                            feature_name = feature
                    else:
                        feature_name = feature
                    print(f"   {i+1:2d}. {feature_name:<25} {score:6.0f}")
        
        # Class distribution analysis
        print(f"\n📊 CLASS DISTRIBUTION ANALYSIS:")
        print("   Actual vs Predicted:")
        actual_dist = pd.Series(y_test).value_counts().sort_index()
        pred_dist = pd.Series(y_pred).value_counts().sort_index()
        
        for class_label, label_name in zip([1, 2, 3, 4], blight_labels):
            actual_count = actual_dist.get(class_label, 0)
            pred_count = pred_dist.get(class_label, 0)
            actual_pct = actual_count / len(y_test) * 100
            pred_pct = pred_count / len(y_pred) * 100
            print(f"   Class {class_label}: Actual {actual_count:,} ({actual_pct:.1f}%) | Predicted {pred_count:,} ({pred_pct:.1f}%)")
        
        # Return metrics for comparison
        return {
            'accuracy': accuracy,
            'balanced_accuracy': balanced_acc,
            'macro_f1': f1_macro,
            'weighted_f1': f1_weighted,
            'log_loss': logloss if 'logloss' in locals() else None,
            'kappa': kappa,
            'confusion_matrix': cm,
            'feature_importance': sorted_importance if 'sorted_importance' in locals() else None,
            'n_samples': len(y_test)
        }
    
    def save_model(self, filepath):
        """Save trained model and metadata."""
        if self.model is None:
            raise ValueError("No model to save. Train a model first.")
            
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'random_state': self.random_state
        }
        
        joblib.dump(model_data, filepath)
        print(f"💾 Model saved to {filepath}")
        
    @classmethod
    def load_model(cls, filepath):
        """Load trained model and metadata."""
        model_data = joblib.load(filepath)
        
        trainer = cls(random_state=model_data.get('random_state', RANDOM_STATE))
        trainer.model = model_data['model']
        trainer.feature_names = model_data['feature_names']
        
        print(f"📂 Model loaded from {filepath}")
        return trainer

def compare_datasets(results_dict):
    """Compare performance across datasets."""
    if len(results_dict) < 2:
        return
        
    print(f"\n{'='*60}")
    print("DATASET COMPARISON SUMMARY")
    print(f"{'='*60}")
    
    metrics = ['accuracy', 'balanced_accuracy', 'macro_f1', 'weighted_f1', 'kappa']
    if all(results['log_loss'] is not None for results in results_dict.values()):
        metrics.append('log_loss')
    
    # Table header
    datasets = list(results_dict.keys())
    print(f"{'Metric':<20} {datasets[0]:<15} {datasets[1]:<15} {'Improvement':<12}")
    print("-" * 65)
    
    for metric in metrics:
        values = [results_dict[dataset][metric] for dataset in datasets]
        
        if metric == 'log_loss':  # Lower is better
            if values[0] > 0:
                improvement = (values[0] - values[1]) / values[0] * 100
            else:
                improvement = 0
        else:  # Higher is better
            if values[0] > 0:
                improvement = (values[1] - values[0]) / values[0] * 100
            else:
                improvement = 0
                
        print(f"{metric:<20} {values[0]:<15.4f} {values[1]:<15.4f} {improvement:+6.1f}%")
    
    # Sample size comparison
    sample_sizes = [results_dict[dataset]['n_samples'] for dataset in datasets]
    size_ratio = sample_sizes[1] / sample_sizes[0] if sample_sizes[0] > 0 else 0
    print(f"\nSample size ratio: {size_ratio:.1f}x larger ({sample_sizes[1]:,} vs {sample_sizes[0]:,})")

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train XGBoost blight detection models')
    parser.add_argument('--data', type=str, help='Single dataset file path')
    parser.add_argument('--small-data', type=str, default='data/blight_data_small.csv', help='Small dataset file path')
    parser.add_argument('--large-data', type=str, default='data/blight_data_large.csv', help='Large dataset file path')
    parser.add_argument('--output-dir', type=str, default='models', 
                       help='Output directory for trained models')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Test set proportion (default: 0.2)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.data and not (args.small_data or args.large_data):
        raise ValueError("Must provide either --data or --small-data/--large-data")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("🏠 BLIGHT DETECTION MODEL TRAINING")
    print("=" * 50)
    print(f"Random seed: {RANDOM_STATE}")
    print(f"Test set size: {args.test_size:.1%}")
    print(f"Output directory: {output_dir}")
    
    results = {}
    
    # Determine datasets to process
    datasets = []
    if args.data:
        datasets.append(("Single Dataset", args.data))
    else:
        if args.small_data:
            datasets.append(("Small Dataset", args.small_data))
        if args.large_data:
            datasets.append(("Large Dataset", args.large_data))
    
    # Process each dataset
    for dataset_name, filepath in datasets:
        print(f"\n{'='*20} PROCESSING {dataset_name.upper()} {'='*20}")
        
        # Initialize trainer
        trainer = BlightModelTrainer(random_state=RANDOM_STATE)
        
        # Load and prepare data
        df = trainer.load_data(filepath)
        X, y = trainer.prepare_features(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=RANDOM_STATE, stratify=y
        )
        
        print(f"   Training set: {len(X_train):,} samples")
        print(f"   Test set: {len(X_test):,} samples")
        
        # Train model
        trainer.train_model(X_train, y_train, X_test, y_test)
        
        # Evaluate model
        eval_results = trainer.evaluate_model(X_test, y_test, dataset_name)
        results[dataset_name] = eval_results
        
        # Save model
        model_filename = f"blight_model_{dataset_name.lower().replace(' ', '_')}.joblib"
        model_path = output_dir / model_filename
        trainer.save_model(model_path)
    
    # Compare datasets if multiple were processed
    if len(results) > 1:
        compare_datasets(results)
    
    print(f"\n🎉 Training complete!")
    print(f"Models saved in: {output_dir}")
    
    return results

if __name__ == "__main__":
    main()