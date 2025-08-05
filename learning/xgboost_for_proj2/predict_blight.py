"""
Blight Detection Prediction Script
=================================

This script loads trained XGBoost models and generates blight predictions
for new parcel data. Supports batch prediction and individual property scoring.

Usage:
    python predict_blight.py --model models/blight_model_large_dataset.joblib --data new_parcels.csv
    python predict_blight.py --model models/blight_model_small_dataset.joblib --parcel-id PARL00001234
"""

import pandas as pd
import numpy as np
import argparse
import joblib
from pathlib import Path
from train_blight_model import BlightModelTrainer

class BlightPredictor:
    """
    Blight prediction interface for trained XGBoost models.
    """
    
    def __init__(self, model_path):
        """Initialize predictor with trained model."""
        self.trainer = BlightModelTrainer.load_model(model_path)
        self.blight_labels = {
            1: 'No Blight',
            2: 'Minor Blight', 
            3: 'Moderate Blight',
            4: 'Severe Blight'
        }
        
    def predict_single(self, parcel_data):
        """
        Predict blight level for a single parcel.
        
        Args:
            parcel_data (dict or pd.Series): Parcel feature data
            
        Returns:
            dict: Prediction results with probabilities
        """
        # Convert to DataFrame if needed
        if isinstance(parcel_data, dict):
            df = pd.DataFrame([parcel_data])
        elif isinstance(parcel_data, pd.Series):
            df = pd.DataFrame([parcel_data])
        else:
            df = parcel_data
            
        # Ensure all required features are present
        missing_features = set(self.trainer.feature_names) - set(df.columns)
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
            
        # Select and order features
        X = df[self.trainer.feature_names]
        
        # Generate predictions
        y_pred, y_pred_proba = self.trainer.predict(X)
        
        # Format results
        result = {
            'predicted_blight_level': int(y_pred[0]),
            'predicted_blight_label': self.blight_labels[y_pred[0]],
            'confidence': float(np.max(y_pred_proba[0])),
            'probabilities': {
                f'level_{i+1}': float(prob) 
                for i, prob in enumerate(y_pred_proba[0])
            }
        }
        
        return result
    
    def predict_batch(self, data_path_or_df):
        """
        Predict blight levels for multiple parcels.
        
        Args:
            data_path_or_df: File path to CSV or DataFrame with parcel data
            
        Returns:
            pd.DataFrame: DataFrame with predictions and probabilities
        """
        # Load data
        if isinstance(data_path_or_df, (str, Path)):
            df = pd.read_csv(data_path_or_df)
            print(f"📂 Loaded {len(df):,} parcels for prediction")
        else:
            df = data_path_or_df.copy()
            
        # Validate required features
        missing_features = set(self.trainer.feature_names) - set(df.columns)
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        # Select features in correct order
        X = df[self.trainer.feature_names]
        
        # Generate predictions
        print("🎯 Generating predictions...")
        y_pred, y_pred_proba = self.trainer.predict(X)
        
        # Add predictions to dataframe
        results_df = df.copy()
        results_df['predicted_blight_level'] = y_pred
        results_df['predicted_blight_label'] = [self.blight_labels[level] for level in y_pred]
        results_df['prediction_confidence'] = np.max(y_pred_proba, axis=1)
        
        # Add probability columns
        for i in range(4):
            results_df[f'prob_level_{i+1}'] = y_pred_proba[:, i]
        
        return results_df
    
    def analyze_predictions(self, predictions_df):
        """Analyze batch prediction results."""
        print(f"\n📊 PREDICTION ANALYSIS")
        print("=" * 40)
        
        # Prediction distribution
        pred_dist = predictions_df['predicted_blight_level'].value_counts().sort_index()
        print(f"Prediction Distribution:")
        for level in [1, 2, 3, 4]:
            count = pred_dist.get(level, 0)
            pct = count / len(predictions_df) * 100
            label = self.blight_labels[level]
            print(f"  Level {level} ({label}): {count:,} ({pct:.1f}%)")
        
        # Confidence analysis
        avg_confidence = predictions_df['prediction_confidence'].mean()
        low_confidence = (predictions_df['prediction_confidence'] < 0.5).sum()
        
        print(f"\nConfidence Analysis:")
        print(f"  Average confidence: {avg_confidence:.3f}")
        print(f"  Low confidence predictions (<0.5): {low_confidence:,} ({low_confidence/len(predictions_df)*100:.1f}%)")
        
        # High-risk properties
        high_risk = predictions_df[predictions_df['predicted_blight_level'].isin([3, 4])]
        print(f"\nHigh-Risk Properties (Levels 3-4): {len(high_risk):,} ({len(high_risk)/len(predictions_df)*100:.1f}%)")
        
        if len(high_risk) > 0:
            print("  Top 5 highest risk properties:")
            top_risk = high_risk.nlargest(5, 'prediction_confidence')
            for _, row in top_risk.iterrows():
                parcel_id = row.get('parcel_id', 'Unknown')
                level = row['predicted_blight_level']
                confidence = row['prediction_confidence']
                print(f"    {parcel_id}: Level {level} (confidence: {confidence:.3f})")

def create_sample_parcel():
    """Create a sample parcel for demonstration."""
    return {
        'property_age': 45,
        'square_footage': 1200,
        'assessed_value': 75000,
        'lot_size': 6000,
        'tax_delinquent_years': 2,
        'tax_amount_owed': 3500,
        'num_code_violations': 3,
        'days_since_last_inspection': 180,
        'open_violations': 1,
        'vacant_mail_holds': 1,
        'delivery_issues': 0,
        'neighborhood_median_income': 35000,
        'crime_incidents_nearby': 4,
        'distance_to_downtown': 5.2,
        'num_foreclosures_nearby': 2,
        'vacant_lots_nearby': 3,
        'utility_shutoffs': 1,
        'water_shutoffs': 0,
        'sidewalk_condition': 2,
        'building_material_quality': 2,
        'roof_condition': 2,
        'lot_condition': 2
    }

def main():
    """Main prediction function."""
    parser = argparse.ArgumentParser(description='Generate blight predictions using trained models')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model file (.joblib)')
    parser.add_argument('--data', type=str,
                       help='CSV file with parcel data for batch prediction')
    parser.add_argument('--output', type=str,
                       help='Output file for batch predictions (default: predictions.csv)')
    parser.add_argument('--parcel-id', type=str,
                       help='Single parcel ID to look up and predict')
    parser.add_argument('--demo', action='store_true',
                       help='Run demo prediction with sample data')
    
    args = parser.parse_args()
    
    # Validate model path
    if not Path(args.model).exists():
        raise FileNotFoundError(f"Model file not found: {args.model}")
    
    print("🏠 BLIGHT DETECTION PREDICTION")
    print("=" * 40)
    print(f"Model: {args.model}")
    
    # Initialize predictor
    predictor = BlightPredictor(args.model)
    print(f"✅ Model loaded successfully")
    print(f"   Features required: {len(predictor.trainer.feature_names)}")
    
    # Demo mode
    if args.demo:
        print(f"\n🎭 DEMO MODE - Sample Prediction")
        print("-" * 30)
        
        sample_parcel = create_sample_parcel()
        result = predictor.predict_single(sample_parcel)
        
        print(f"Sample parcel features:")
        for key, value in list(sample_parcel.items())[:5]:  # Show first 5 features
            print(f"  {key}: {value}")
        print(f"  ... and {len(sample_parcel)-5} more features")
        
        print(f"\n🎯 Prediction Results:")
        print(f"  Predicted Level: {result['predicted_blight_level']}")
        print(f"  Predicted Label: {result['predicted_blight_label']}")
        print(f"  Confidence: {result['confidence']:.3f}")
        
        print(f"\n📊 Class Probabilities:")
        for level, prob in result['probabilities'].items():
            level_num = int(level.split('_')[1])
            label = predictor.blight_labels[level_num]
            print(f"  {label}: {prob:.3f}")
    
    # Single parcel prediction
    elif args.parcel_id:
        print(f"\n🔍 SINGLE PARCEL PREDICTION")
        print(f"Parcel ID: {args.parcel_id}")
        print("Note: This is a demo - in practice, you would look up parcel data from a database")
        
        # For demo, use sample data
        sample_parcel = create_sample_parcel()
        sample_parcel['parcel_id'] = args.parcel_id
        
        result = predictor.predict_single(sample_parcel)
        
        print(f"\n🎯 Prediction Results:")
        print(f"  Predicted Level: {result['predicted_blight_level']}")
        print(f"  Predicted Label: {result['predicted_blight_label']}")
        print(f"  Confidence: {result['confidence']:.3f}")
    
    # Batch prediction
    elif args.data:
        print(f"\n📊 BATCH PREDICTION")
        print(f"Input data: {args.data}")
        
        if not Path(args.data).exists():
            raise FileNotFoundError(f"Data file not found: {args.data}")
        
        # Generate predictions
        predictions_df = predictor.predict_batch(args.data)
        
        # Analyze results
        predictor.analyze_predictions(predictions_df)
        
        # Save results
        output_file = args.output or 'predictions.csv'
        predictions_df.to_csv(output_file, index=False)
        print(f"\n💾 Predictions saved to: {output_file}")
        
        # Show sample results
        print(f"\n📋 Sample Predictions (first 5 rows):")
        display_cols = ['parcel_id', 'predicted_blight_level', 'predicted_blight_label', 'prediction_confidence']
        available_cols = [col for col in display_cols if col in predictions_df.columns]
        
        if available_cols:
            sample_results = predictions_df[available_cols].head()
            print(sample_results.to_string(index=False))
    
    else:
        print("\n❌ No prediction mode specified.")
        print("Use --demo, --data <file>, or --parcel-id <id>")
        return
    
    print(f"\n🎉 Prediction complete!")

if __name__ == "__main__":
    main()