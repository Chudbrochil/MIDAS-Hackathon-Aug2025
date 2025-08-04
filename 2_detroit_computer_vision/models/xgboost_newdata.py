import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                             f1_score, cohen_kappa_score, matthews_corrcoef,
                             balanced_accuracy_score)
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime

# Note: Feature columns are now determined from the input CSV files
# The features file should contain all predictive columns
# The labels file should contain PARCEL_ID_x, label (0-3), and label_str

# TARGET COLUMN
TARGET_COLUMN = 'label'

# PARCEL ID COLUMN (for tracking)
ID_COLUMN = 'PARCEL_ID_x'

def load_data(features_path, labels_path):
    """Load the features and labels from separate CSV files."""
    print("Loading data...")
    features_df = pd.read_csv(features_path, dtype={'PARCEL_ID_x': str})
    labels_df = pd.read_csv(labels_path, dtype={'PARCEL_ID_x': str})
    
    # Drop any unnamed columns first
    features_df = features_df.loc[:, ~features_df.columns.str.contains('^Unnamed')]
    labels_df = labels_df.loc[:, ~labels_df.columns.str.contains('^Unnamed')]
    
    print(f"Original features: {len(features_df)}, labels: {len(labels_df)}")
    
    # Remove duplicates - keep first occurrence only
    print("Removing duplicates...")
    features_df = features_df.drop_duplicates(subset=['PARCEL_ID_x'], keep='first')
    labels_df = labels_df.drop_duplicates(subset=['PARCEL_ID_x'], keep='first')
    
    print(f"After deduplication - features: {len(features_df)}, labels: {len(labels_df)}")
    
    # Merge on PARCEL_ID_x
    data = features_df.merge(labels_df, on='PARCEL_ID_x', how='inner')
    
    print(f"After merge: {len(data)} samples")
    
    # Verify no duplicates in final dataset
    if data['PARCEL_ID_x'].duplicated().sum() > 0:
        print(f"WARNING: {data['PARCEL_ID_x'].duplicated().sum()} duplicate PARCEL_ID_x in final dataset")
        data = data.drop_duplicates(subset=['PARCEL_ID_x'], keep='first')
        print(f"After final deduplication: {len(data)} samples")
    
    return data

def prepare_features(data):
    """Prepare features for training by encoding categorical variables."""
    # Get all columns except ID, target, and label_str
    feature_cols = [col for col in data.columns if col not in [ID_COLUMN, TARGET_COLUMN, 'label_str']]
    features_df = data[feature_cols].copy()
    
    # Identify categorical columns
    categorical_cols = []
    for col in features_df.columns:
        if features_df[col].dtype == 'object' or col in ['IS_OCCUPIED', 'HAS_STRUCTURE',
                                                          'VACANT', 'account_active_code', 
                                                          'tax_status_x', 'is_improved_x',
                                                          'property_class_desc', 'SURVEYOR_NOTES',
                                                          'permit_type', 'local_historic_district']:
            categorical_cols.append(col)
    
    print(f"\nCategorical columns to encode: {categorical_cols}")
    
    # Encode categorical variables
    label_encoders = {}
    for col in categorical_cols:
        if col in features_df.columns:
            le = LabelEncoder()
            # Convert to string to handle mixed types and fill NaN
            features_df[col] = features_df[col].fillna('Unknown').astype(str)
            features_df[col] = le.fit_transform(features_df[col])
            label_encoders[col] = le
            print(f"  Encoded {col}: {len(le.classes_)} unique values")
    
    # Handle numeric columns - fill NaN with median
    numeric_cols = [col for col in features_df.columns if col not in categorical_cols]
    for col in numeric_cols:
        if features_df[col].dtype in ['float64', 'int64']:
            median_val = features_df[col].median()
            features_df[col] = features_df[col].fillna(median_val)
            print(f"  Filled NaN in {col} with median: {median_val}")
    
    return features_df, label_encoders

def prepare_target(data):
    """Prepare target variable for training."""
    # Target is already numeric (0-3), no encoding needed
    y = data[TARGET_COLUMN].copy()
    
    # Get label names if available - handle multiple strings per numeric label
    label_names = ['Class 0', 'Class 1', 'Class 2', 'Class 3']  # Default
    if 'label_str' in data.columns:
        # Get the most common string for each numeric label
        label_mapping = data[[TARGET_COLUMN, 'label_str']].groupby(TARGET_COLUMN)['label_str'].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else 'Unknown').sort_index()
        
        # Ensure we have exactly 4 labels (0, 1, 2, 3)
        label_names = []
        for i in range(4):
            if i in label_mapping.index:
                label_names.append(label_mapping[i])
            else:
                label_names.append(f'Class {i}')
        
        print(f"\nLabel mapping found:")
        for cls, name in enumerate(label_names):
            print(f"  {cls}: {name}")
    
    # Create a pseudo label encoder for compatibility
    class PseudoLabelEncoder:
        def __init__(self, classes):
            self.classes_ = np.array(classes)
    
    le_target = PseudoLabelEncoder(label_names)
    
    print(f"\nTarget classes: {label_names}")
    print(f"Class distribution:")
    unique, counts = np.unique(y, return_counts=True)
    for cls, count in zip(unique, counts):
        if cls < len(label_names):
            print(f"  Class {cls} ({label_names[cls]}): {count} ({count/len(y)*100:.2f}%)")
    
    return y, le_target

def analyze_data(data, y_encoded, le_target, output_dir='../deliverables/xgboost_newdata'):
    """Perform basic analytics on the dataset and save results."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    analytics = {}
    
    # Total samples
    analytics['total_samples'] = len(data)
    
    # Label distribution
    unique, counts = np.unique(y_encoded, return_counts=True)
    label_dist = {int(k): int(v) for k, v in zip(unique, counts)}
    analytics['label_distribution'] = label_dist
    analytics['label_names'] = {int(i): le_target.classes_[i] for i in unique}
    analytics['label_percentages'] = {int(i): float(count/len(y_encoded)*100) for i, count in label_dist.items()}
    
    # Feature statistics
    feature_cols = [col for col in data.columns if col not in [ID_COLUMN, TARGET_COLUMN, 'label_str']]
    analytics['feature_count'] = len(feature_cols)
    analytics['features'] = feature_cols
    
    # Save analytics to file
    with open(f'{output_dir}/data_analytics.json', 'w') as f:
        json.dump(analytics, f, indent=2)
    
    # Create label distribution plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Subplot 1: Count distribution
    bars = ax1.bar(label_dist.keys(), label_dist.values())
    ax1.set_xlabel('Class')
    ax1.set_ylabel('Count')
    ax1.set_title('Distribution of FIELD_DETERMINATION (Counts)')
    ax1.set_xticks(list(label_dist.keys()))
    
    # Add count labels on bars
    for bar, (cls, count) in zip(bars, label_dist.items()):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 100,
                f'{count:,}', ha='center', va='bottom')
    
    # Subplot 2: Percentage distribution (pie chart)
    percentages = [analytics['label_percentages'][i] for i in sorted(analytics['label_percentages'].keys())]
    labels = [f'{le_target.classes_[i]}\n({pct:.1f}%)' 
              for i, pct in sorted(analytics['label_percentages'].items())]
    ax2.pie(percentages, labels=labels, autopct='%1.1f%%', startangle=90)
    ax2.set_title('Distribution of FIELD_DETERMINATION (Percentages)')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/label_distribution.png', dpi=300)
    plt.close()
    
    print(f"\nData Analytics:")
    print(f"Total samples: {analytics['total_samples']}")
    print("\nLabel distribution:")
    for cls, count in analytics['label_distribution'].items():
        pct = analytics['label_percentages'][cls]
        name = analytics['label_names'][cls]
        print(f"  Class {cls} ({name}): {count} ({pct:.2f}%)")
    
    return analytics

def split_data(X, y, parcel_ids=None, test_size=0.2, val_size=0.1, random_state=42):
    """Split data into train, validation, and test sets with optional parcel ID tracking."""
    # Create indices for tracking
    indices = np.arange(len(X))
    
    # First split: train+val vs test
    if parcel_ids is not None:
        X_temp, X_test, y_temp, y_test, idx_temp, idx_test = train_test_split(
            X, y, indices, test_size=test_size, random_state=random_state, stratify=y
        )
    else:
        X_temp, X_test, y_temp, y_test, idx_temp, idx_test = train_test_split(
            X, y, indices, test_size=test_size, random_state=random_state, stratify=y
        )
    
    # Second split: train vs val
    val_size_adjusted = val_size / (1 - test_size)  # Adjust val size for remaining data
    X_train, X_val, y_train, y_val, idx_train, idx_val = train_test_split(
        X_temp, y_temp, idx_temp, test_size=val_size_adjusted, random_state=random_state, stratify=y_temp
    )
    
    print(f"\nData split:")
    print(f"  Training samples: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  Validation samples: {len(X_val)} ({len(X_val)/len(X)*100:.1f}%)")
    print(f"  Test samples: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")
    
    # Return indices along with data if parcel_ids provided
    if parcel_ids is not None:
        parcel_ids_train = parcel_ids.iloc[idx_train]
        parcel_ids_val = parcel_ids.iloc[idx_val]
        parcel_ids_test = parcel_ids.iloc[idx_test]
        return (X_train, X_val, X_test, y_train, y_val, y_test, 
                parcel_ids_train, parcel_ids_val, parcel_ids_test)
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def train_model(X_train, y_train, X_val=None, y_val=None, params=None):
    """Train XGBoost model with given parameters."""
    if params is None:
        # Determine number of classes
        n_classes = len(np.unique(y_train))
        
        params = {
            'objective': 'multi:softprob',
            'num_class': n_classes,
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'random_state': 42,
            'eval_metric': 'mlogloss'
        }
    
    print("\nTraining XGBoost model...")
    print(f"Number of classes: {params['num_class']}")
    model = xgb.XGBClassifier(**params)
    
    # Prepare eval set if validation data provided
    eval_set = [(X_train, y_train)]
    if X_val is not None and y_val is not None:
        eval_set.append((X_val, y_val))
    
    model.fit(
        X_train, y_train,
        eval_set=eval_set,
        verbose=False
    )
    
    return model

def predict(model, X):
    """Make predictions with the trained model."""
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)
    
    return y_pred, y_pred_proba

def save_predictions_with_ids(y_pred, y_true, parcel_ids, le_target, y_pred_proba=None, 
                              output_path='../deliverables/xgboost_newdata/predictions.csv'):
    """Save predictions along with parcel IDs for traceability."""
    # Create predictions dataframe
    predictions_df = pd.DataFrame({
        ID_COLUMN: parcel_ids,
        'true_label': y_true,
        'true_label_name': [le_target.classes_[i] for i in y_true],
        'predicted_label': y_pred,
        'predicted_label_name': [le_target.classes_[i] for i in y_pred]
    })
    
    # Add probability columns if provided
    if y_pred_proba is not None:
        for i in range(y_pred_proba.shape[1]):
            predictions_df[f'prob_class_{i}_{le_target.classes_[i]}'] = y_pred_proba[:, i]
    
    # Save to CSV
    predictions_df.to_csv(output_path, index=False)
    print(f"\nPredictions saved to: {output_path}")
    
    return predictions_df

def evaluate_model(model, X_test, y_test, le_target, output_dir='../deliverables/xgboost_newdata'):
    """Evaluate model performance with multiple metrics."""
    # Make predictions
    y_pred, y_pred_proba = predict(model, X_test)
    
    # Calculate various metrics
    accuracy = accuracy_score(y_test, y_pred)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    weighted_f1 = f1_score(y_test, y_pred, average='weighted')
    kappa = cohen_kappa_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    
    # Per-class F1 scores
    per_class_f1 = f1_score(y_test, y_pred, average=None)
    
    # Full classification report - don't use target_names to avoid mismatch
    report = classification_report(y_test, y_pred, output_dict=True)
    
    print(f"\n{'='*50}")
    print("MODEL PERFORMANCE METRICS")
    print(f"{'='*50}")
    print(f"\nOverall Metrics:")
    print(f"  Accuracy: {accuracy:.3f}")
    print(f"  Balanced Accuracy: {balanced_acc:.3f}")
    print(f"  Macro F1-score: {macro_f1:.3f}")
    print(f"  Weighted F1-score: {weighted_f1:.3f}")
    print(f"  Cohen's Kappa: {kappa:.3f}")
    print(f"  MCC: {mcc:.3f}")
    
    print(f"\nPer-Class F1-scores:")
    for i, f1 in enumerate(per_class_f1):
        if i < len(le_target.classes_):
            class_name = le_target.classes_[i]
        else:
            class_name = f"Class {i}"
        
        # Get support from report using string key
        class_support = report.get(str(i), {}).get('support', 0)
        print(f"  Class {i} ({class_name}): {f1:.3f} (n={class_support})")
    
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save evaluation results
    evaluation_results = {
        'accuracy': accuracy,
        'balanced_accuracy': balanced_acc,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'cohens_kappa': kappa,
        'mcc': mcc,
        'per_class_f1': per_class_f1.tolist(),
        'class_names': le_target.classes_.tolist(),
        'classification_report': report,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(f'{output_dir}/evaluation_results.json', 'w') as f:
        json.dump(evaluation_results, f, indent=2)
    
    # Create metrics comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Overall metrics bar plot
    metrics_names = ['Accuracy', 'Balanced\nAccuracy', 'Macro F1', 'Weighted F1', "Cohen's\nKappa", 'MCC']
    metrics_values = [accuracy, balanced_acc, macro_f1, weighted_f1, kappa, mcc]
    
    bars = ax1.bar(metrics_names, metrics_values, color=['blue', 'green', 'orange', 'red', 'purple', 'brown'])
    ax1.set_ylim(0, 1)
    ax1.set_ylabel('Score')
    ax1.set_title('Overall Model Performance Metrics')
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    
    # Add value labels on bars
    for bar, val in zip(bars, metrics_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{val:.3f}', ha='center', va='bottom')
    
    # Per-class F1 scores
    class_labels = [f'{le_target.classes_[i] if i < len(le_target.classes_) else f"Class {i}"}' for i in range(len(per_class_f1))]
    bars2 = ax2.bar(range(len(per_class_f1)), per_class_f1)
    ax2.set_ylim(0, 1)
    ax2.set_ylabel('F1-Score')
    ax2.set_title('Per-Class F1-Scores')
    ax2.set_xticks(range(len(per_class_f1)))
    ax2.set_xticklabels(class_labels, rotation=45, ha='right')
    
    # Add value and support labels
    for i, (bar, f1) in enumerate(zip(bars2, per_class_f1)):
        height = bar.get_height()
        support = report.get(str(i), {}).get('support', 0)
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{f1:.3f}\n(n={support})', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/metrics_comparison.png', dpi=300)
    plt.close()
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=le_target.classes_, yticklabels=le_target.classes_)
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual')
    ax1.set_title('Confusion Matrix (Counts)')
    
    # Normalized (row-wise percentages)
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues', ax=ax2,
                xticklabels=le_target.classes_, yticklabels=le_target.classes_)
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('Actual')
    ax2.set_title('Confusion Matrix (Row-normalized %)')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/confusion_matrix.png', dpi=300)
    plt.close()
    
    return evaluation_results

def analyze_feature_importance(model, feature_names, output_dir='../deliverables/xgboost_newdata'):
    """Analyze and visualize feature importance."""
    # Get feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Feature Importances:")
    print(feature_importance.head(10))
    
    # Save feature importance
    feature_importance.to_csv(f'{output_dir}/feature_importance.csv', index=False)
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance['feature'][:15], feature_importance['importance'][:15])
    plt.xlabel('Importance')
    plt.title('XGBoost Feature Importances (Top 15)')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/feature_importance.png', dpi=300)
    plt.close()
    
    return feature_importance

def main():
    """Main training pipeline."""
    # Paths to the new cleaned features and labels files
    features_path = "../training_data/08042025-cleaned-features.csv"
    labels_path = "../training_data/08042025-cleaned-labels.csv"
    
    # Load data
    data = load_data(features_path, labels_path)
    
    # Prepare target variable
    y_encoded, le_target = prepare_target(data)
    
    # Analyze data
    analytics = analyze_data(data, y_encoded, le_target)
    
    # Extract parcel IDs
    parcel_ids = data[ID_COLUMN]
    
    # Prepare features
    X, label_encoders = prepare_features(data)
    
    # Split data with parcel ID tracking
    split_results = split_data(X, y_encoded, parcel_ids)
    X_train, X_val, X_test = split_results[0:3]
    y_train, y_val, y_test = split_results[3:6]
    parcel_ids_train, parcel_ids_val, parcel_ids_test = split_results[6:9]
    
    # Train model
    model = train_model(X_train, y_train, X_val, y_val)
    
    # Evaluate model
    evaluation_results = evaluate_model(model, X_test, y_test, le_target)
    
    # Make predictions and save with parcel IDs
    y_pred_test, y_pred_proba_test = predict(model, X_test)
    predictions_df = save_predictions_with_ids(
        y_pred_test, y_test, parcel_ids_test, le_target,
        y_pred_proba_test, '../deliverables/xgboost_newdata/test_predictions.csv'
    )
    
    # Analyze feature importance
    feature_importance = analyze_feature_importance(model, X.columns.tolist())
    
    return model, analytics, feature_importance, predictions_df, le_target

if __name__ == "__main__":
    model, analytics, feature_importance, predictions_df, le_target = main()