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

def load_data(features_path, labels_path):
    """Load and merge feature and label data."""
    print("Loading data...")
    features_df = pd.read_csv(features_path)
    labels_df = pd.read_csv(labels_path)
    
    # Merge on PARCEL_ID
    data = features_df.merge(labels_df, on='PARCEL_ID', how='inner')
    
    # Drop any unnamed columns
    data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
    
    return data

def prepare_features(features_df):
    """Prepare features for training by encoding categorical variables."""
    df = features_df.copy()
    
    # Drop DATE_SURVEYED - not likely to be predictive
    if 'DATE_SURVEYED' in df.columns:
        df = df.drop('DATE_SURVEYED', axis=1)
    
    # Encode categorical variables
    categorical_cols = ['IS_OCCUPIED', 'FIRE_DAMAGE_CONDITION', 'ROOF_CONDITION', 
                       'OPENINGS_CONDITION', 'IS_OPEN_TO_TRESPASS']
    
    label_encoders = {}
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            # Convert to string to handle mixed types
            df[col] = df[col].fillna('Unknown').astype(str)
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le
    
    return df, label_encoders

def analyze_data(data, output_dir='../deliverables/xgboost_baseline'):
    """Perform basic analytics on the dataset and save results."""
    analytics = {}
    
    # Total samples
    analytics['total_samples'] = len(data)
    
    # Label distribution
    label_dist = data['BLIGHT_LABEL'].value_counts().sort_index()
    analytics['label_distribution'] = label_dist.to_dict()
    analytics['label_percentages'] = (label_dist / len(data) * 100).round(2).to_dict()
    
    # Feature statistics
    feature_cols = [col for col in data.columns if col not in ['PARCEL_ID', 'BLIGHT_LABEL']]
    analytics['feature_count'] = len(feature_cols)
    analytics['features'] = feature_cols
    
    # Save analytics to file
    with open(f'{output_dir}/data_analytics.json', 'w') as f:
        json.dump(analytics, f, indent=2)
    
    # Create combined label distribution plot (counts and percentages)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Subplot 1: Count distribution
    colors = ['green', 'yellow', 'orange', 'red']
    bars = ax1.bar(label_dist.index, label_dist.values, color=colors)
    ax1.set_xlabel('Blight Label')
    ax1.set_ylabel('Count')
    ax1.set_title('Distribution of Blight Labels (Counts)')
    ax1.set_xticks(label_dist.index)
    
    # Add count labels on bars
    for bar, count in zip(bars, label_dist.values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 100,
                f'{count:,}', ha='center', va='bottom')
    
    # Subplot 2: Percentage distribution (pie chart)
    percentages = [analytics['label_percentages'][i] for i in sorted(analytics['label_percentages'].keys())]
    labels = [f'Label {i}\n({pct}%)' for i, pct in zip(label_dist.index, percentages)]
    ax2.pie(percentages, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax2.set_title('Distribution of Blight Labels (Percentages)')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/label_distribution.png', dpi=300)
    plt.close()
    
    # Create summary statistics plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Summary data
    summary_data = {
        'Total Samples': analytics['total_samples'],
        'Number of Features': analytics['feature_count'],
        'Class 0 (No Blight)': analytics['label_distribution'][0],
        'Class 1 (Noticeable)': analytics['label_distribution'][1],
        'Class 2 (Significant)': analytics['label_distribution'][2],
        'Class 3 (Extreme)': analytics['label_distribution'][3]
    }
    
    # Create text plot
    ax.axis('off')
    y_pos = 0.9
    for key, value in summary_data.items():
        ax.text(0.1, y_pos, f'{key}:', fontsize=12, fontweight='bold')
        ax.text(0.6, y_pos, f'{value:,}', fontsize=12)
        y_pos -= 0.12
    
    ax.set_title('Dataset Summary Statistics', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/data_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nData Analytics:")
    print(f"Total samples: {analytics['total_samples']}")
    print("\nLabel distribution:")
    for label, count in analytics['label_distribution'].items():
        pct = analytics['label_percentages'][label]
        print(f"  Label {label}: {count} ({pct}%)")
    
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

def split_data_kfold(X, y, n_splits=5, random_state=42):
    """Prepare K-fold cross-validation splits."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    print(f"\nPrepared {n_splits}-fold cross-validation")
    
    # Return the KFold object for later use
    return kf

def train_model(X_train, y_train, X_val=None, y_val=None, params=None):
    """Train XGBoost model with given parameters."""
    if params is None:
        params = {
            'objective': 'multi:softprob',
            'num_class': 4,
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'random_state': 42,
            'eval_metric': 'mlogloss'
        }
    
    print("\nTraining XGBoost model...")
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

def save_predictions_with_ids(y_pred, y_true, parcel_ids, y_pred_proba=None, output_path='../deliverables/xgboost_baseline/predictions.csv'):
    """Save predictions along with PARCEL_IDs for traceability."""
    # Create predictions dataframe
    predictions_df = pd.DataFrame({
        'PARCEL_ID': parcel_ids,
        'true_label': y_true,
        'predicted_label': y_pred
    })
    
    # Add probability columns if provided
    if y_pred_proba is not None:
        for i in range(y_pred_proba.shape[1]):
            predictions_df[f'prob_class_{i}'] = y_pred_proba[:, i]
    
    # Save to CSV
    predictions_df.to_csv(output_path, index=False)
    print(f"\nPredictions saved to: {output_path}")
    
    return predictions_df

def evaluate_model(model, X_test, y_test, output_dir='../deliverables/xgboost_baseline'):
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
    
    # Full classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    
    print(f"\n{'='*50}")
    print("MODEL PERFORMANCE METRICS")
    print(f"{'='*50}")
    print(f"\nOverall Metrics:")
    print(f"  Accuracy: {accuracy:.3f} (can be misleading with imbalanced classes)")
    print(f"  Balanced Accuracy: {balanced_acc:.3f} (better for imbalanced)")
    print(f"  Macro F1-score: {macro_f1:.3f} (treats all classes equally)")
    print(f"  Weighted F1-score: {weighted_f1:.3f} (weighted by class frequency)")
    print(f"  Cohen's Kappa: {kappa:.3f} (agreement corrected for chance)")
    print(f"  MCC: {mcc:.3f} (correlation coefficient)")
    
    print(f"\nPer-Class F1-scores:")
    for i, f1 in enumerate(per_class_f1):
        class_support = report[str(i)]['support']
        print(f"  Class {i}: {f1:.3f} (n={class_support})")
    
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
    class_labels = [f'Class {i}' for i in range(len(per_class_f1))]
    colors = ['green', 'yellow', 'orange', 'red']
    bars2 = ax2.bar(class_labels, per_class_f1, color=colors[:len(per_class_f1)])
    ax2.set_ylim(0, 1)
    ax2.set_ylabel('F1-Score')
    ax2.set_title('Per-Class F1-Scores')
    
    # Add value and support labels
    for i, (bar, f1) in enumerate(zip(bars2, per_class_f1)):
        height = bar.get_height()
        support = report[str(i)]['support']
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{f1:.3f}\n(n={support})', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/metrics_comparison.png', dpi=300)
    plt.close()
    
    # Confusion matrix with normalized version
    cm = confusion_matrix(y_test, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1)
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual')
    ax1.set_title('Confusion Matrix (Counts)')
    
    # Normalized (row-wise percentages)
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues', ax=ax2)
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('Actual')
    ax2.set_title('Confusion Matrix (Row-normalized %)')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/confusion_matrix.png', dpi=300)
    plt.close()
    
    return evaluation_results

def analyze_feature_importance(model, feature_names, output_dir='../deliverables/xgboost_baseline'):
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
    # Paths
    features_path = "../training_data/blight_features.csv"
    labels_path = "../training_data/blight_labels.csv"
    
    # Load data
    data = load_data(features_path, labels_path)
    
    # Analyze data
    analytics = analyze_data(data)
    
    # Extract parcel IDs before preprocessing
    parcel_ids = data['PARCEL_ID']
    
    # Prepare features
    X, label_encoders = prepare_features(data.drop(['PARCEL_ID', 'BLIGHT_LABEL'], axis=1))
    y = data['BLIGHT_LABEL']
    
    # Split data with parcel ID tracking
    split_results = split_data(X, y, parcel_ids)
    X_train, X_val, X_test = split_results[0:3]
    y_train, y_val, y_test = split_results[3:6]
    parcel_ids_train, parcel_ids_val, parcel_ids_test = split_results[6:9]
    
    # Train model
    model = train_model(X_train, y_train, X_val, y_val)
    
    # Evaluate model
    evaluation_results = evaluate_model(model, X_test, y_test)
    
    # Make predictions and save with PARCEL_IDs
    y_pred_test, y_pred_proba_test = predict(model, X_test)
    predictions_df = save_predictions_with_ids(
        y_pred_test, y_test, parcel_ids_test, 
        y_pred_proba_test, '../deliverables/xgboost_baseline/test_predictions.csv'
    )
    
    # Analyze feature importance
    feature_importance = analyze_feature_importance(model, X.columns.tolist())
    
    # Optional: K-fold cross-validation
    # kf = split_data_kfold(X, y)
    # for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    #     X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
    #     y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
    #     model_fold = train_model(X_train_fold, y_train_fold)
    #     # Evaluate on fold...
    
    return model, analytics, feature_importance, predictions_df

if __name__ == "__main__":
    model, analytics, feature_importance, predictions_df = main()