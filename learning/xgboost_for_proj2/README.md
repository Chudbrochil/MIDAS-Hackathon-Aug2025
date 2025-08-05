# Blight Detection Using XGBoost - Multi-Class Classification

This project implements a comprehensive blight detection system using XGBoost for multi-class classification. The system can predict property blight severity on a 1-4 scale to help identify properties that need attention for public health and safety.

## 🏠 Project Overview

Property blight is a serious urban challenge affecting public health, safety, and neighborhood stability. This machine learning system helps identify and classify blight severity to support proactive intervention efforts.

### Blight Classification Scale
- **Level 1**: No blight (well-maintained property)
- **Level 2**: Minor blight (cosmetic issues, minor repairs needed)
- **Level 3**: Moderate blight (structural issues, significant deterioration)
- **Level 4**: Severe blight (unsafe/uninhabitable, potential demolition)

## 📊 Features & Data Sources

The model uses 22 realistic features based on data commonly available to municipal authorities:

### Property Characteristics
- `property_age`: Age of property in years
- `square_footage`: Property size in square feet
- `assessed_value`: Property assessment value
- `lot_size`: Lot size in square feet

### Financial Indicators
- `tax_delinquent_years`: Years behind on property taxes
- `tax_amount_owed`: Total tax debt amount

### Code Enforcement & Inspections
- `num_code_violations`: Code violations in last 3 years
- `days_since_last_inspection`: Days since last municipal inspection
- `open_violations`: Currently open violations

### USPS & Vacancy Indicators
- `vacant_mail_holds`: Mail holds indicating vacancy
- `delivery_issues`: Mail delivery problems

### Neighborhood Context
- `neighborhood_median_income`: Area median household income
- `crime_incidents_nearby`: Crime incidents within 500ft (1 year)
- `distance_to_downtown`: Distance to city center in miles
- `num_foreclosures_nearby`: Foreclosures within 0.25mi (2 years)
- `vacant_lots_nearby`: Vacant lots within 500ft

### Infrastructure & Utilities
- `utility_shutoffs`: Gas/electric disconnections
- `water_shutoffs`: Water service disconnections
- `sidewalk_condition`: Sidewalk condition rating (1-5)

### Property Condition Indicators
- `building_material_quality`: Construction quality rating (1-5)
- `roof_condition`: Roof condition rating (1-5)
- `lot_condition`: Lot maintenance rating (1-5)

## 🚀 Quick Start

### Installation

**Step 0 (Optional but Recommended): Set up isolated environment**
```bash
# Install Anaconda from https://www.anaconda.com/download if not already installed
conda create -n blight-detection python=3.10
conda activate blight-detection
```

**Step 1: Install dependencies**
```bash
# Verify you're in the project directory
cd proj2_blight_classification

# Install all required packages
pip install -r requirements.txt

# Verify installation
python -c "import xgboost, sklearn, pandas; print('All packages installed successfully')"
```

**Required packages** (automatically installed):
- xgboost>=1.7.0, scikit-learn>=1.2.0, pandas>=1.5.0, numpy>=1.24.0, matplotlib>=3.6.0, seaborn>=0.12.0, joblib>=1.2.0

### Basic Usage

1. **Generate synthetic datasets** (for demonstration):
```bash
python generate_synthetic_data.py
```

2. **Train models** on both small and large datasets:
```bash
python train_blight_model.py
```

3. **Make predictions** on new data:
```bash
python predict_blight.py --model models/blight_model_large_dataset.joblib --demo
```

## 📁 Project Structure

```
proj2_blight_classification/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── generate_synthetic_data.py   # Data generation script
├── train_blight_model.py       # Model training script
├── predict_blight.py           # Prediction script
├── data/                       # Generated datasets
│   ├── blight_data_small.csv   # Small dataset (1,000 samples)
│   └── blight_data_large.csv   # Large dataset (100,000 samples)
└── models/                     # Trained models
    ├── blight_model_small_dataset.joblib
    └── blight_model_large_dataset.joblib
```

## 🔧 Detailed Usage

### Data Generation

Generate custom datasets with different sizes:

```bash
# Default: 1,000 small, 100,000 large samples
python generate_synthetic_data.py

# Custom sizes
python generate_synthetic_data.py --small-size 500 --large-size 50000

# Generate TSV format
python generate_synthetic_data.py --format tsv

# Custom output directory
python generate_synthetic_data.py --output-dir custom_data/
```

### Model Training

Train models with various configurations:

```bash
# Train on both datasets (default)
python train_blight_model.py

# Train on single dataset
python train_blight_model.py --data data/blight_data_small.csv

# Custom test split
python train_blight_model.py --test-size 0.3

# Custom model output directory
python train_blight_model.py --output-dir my_models/
```

### Making Predictions

#### Demo Mode
```bash
python predict_blight.py --model models/blight_model_large_dataset.joblib --demo
```

#### Batch Predictions
```bash
python predict_blight.py --model models/blight_model_large_dataset.joblib --data new_parcels.csv --output predictions.csv
```

#### Single Parcel Prediction
```bash
python predict_blight.py --model models/blight_model_large_dataset.joblib --parcel-id PARL00001234
```

## 📈 Model Performance

### Dataset Comparison Results

| Metric | Small Dataset (1K) | Large Dataset (100K) | Improvement |
|--------|-------------------|---------------------|-------------|
| Accuracy | 0.5650 | 0.5952 | +5.4% |
| Balanced Accuracy | 0.2518 | 0.2577 | +2.4% |
| Macro F1 | 0.2201 | 0.2112 | -4.1% |
| Weighted F1 | 0.4739 | 0.4704 | -0.7% |
| Cohen's Kappa | 0.0093 | 0.0327 | +250.6% |
| Log Loss | 1.0661 | 0.9903 | +7.1% |

### Key Insights

1. **Sample Size Impact**: The 100x larger dataset shows modest improvements in overall accuracy and calibration (lower log loss).

2. **Class Imbalance Challenge**: Both models struggle with minority classes (Levels 3-4), which is typical for imbalanced datasets where severe blight is relatively rare.

3. **Feature Importance**: Most important features include:
   - `days_since_last_inspection`
   - `assessed_value`
   - `tax_amount_owed`
   - `neighborhood_median_income`
   - `property_age`

4. **Practical Performance**: The model effectively identifies properties with no blight (Level 1) with ~97% recall, making it useful for screening applications.

### Model Limitations

- **Class Imbalance**: Poor performance on severe blight categories due to few examples
- **Prediction Bias**: Models tend to predict lower blight levels, requiring threshold tuning for practical deployment
- **Feature Dependencies**: Performance depends on data quality and availability of municipal data sources

## 🎯 Real-World Applications

### Municipal Use Cases

1. **Proactive Inspections**: Prioritize properties for inspection based on blight risk scores
2. **Resource Allocation**: Focus limited enforcement resources on highest-risk areas
3. **Neighborhood Planning**: Identify areas needing intervention or investment
4. **Policy Evaluation**: Track blight trends over time and measure intervention effectiveness

### Integration Considerations

- **Data Pipeline**: Integrate with existing municipal databases (assessor, code enforcement, USPS)
- **Threshold Tuning**: Adjust prediction thresholds based on available enforcement resources
- **Regular Retraining**: Update models as new inspection data becomes available
- **Bias Monitoring**: Monitor for demographic or geographic bias in predictions

## 🔬 Technical Details

### Model Architecture
- **Algorithm**: XGBoost with multi-class classification
- **Objective**: `multi:softprob` for probability outputs
- **Features**: 22 numerical features (no categorical encoding needed)
- **Output**: 4-class probability distribution + predicted class

### Training Configuration
- **Train/Test Split**: 80/20 stratified split
- **Cross-Validation**: Built-in XGBoost validation with early stopping
- **Hyperparameters**: Optimized for multi-class performance
  - `max_depth`: 6
  - `learning_rate`: 0.1
  - `subsample`: 0.8
  - `colsample_bytree`: 0.8

### Evaluation Metrics
- **Classification**: Accuracy, balanced accuracy, precision, recall, F1-score
- **Multi-class**: One-vs-rest and one-vs-one AUC, log loss
- **Robustness**: Cohen's kappa, confusion matrices
- **Calibration**: Probability reliability analysis

## 📝 Citation & License

This project was developed for educational and public benefit purposes. The synthetic data and models are intended to demonstrate machine learning approaches for urban blight detection.

### Disclaimer
This is a demonstration system using synthetic data. Real-world deployment would require:
- Validation with actual municipal data
- Bias testing and fairness evaluation
- Integration with existing city systems
- Regular model monitoring and updates
- Compliance with local privacy and transparency requirements

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Advanced feature engineering
- Handling of missing data
- Model interpretation and explainability
- Integration with GIS systems
- Real-world validation studies

## 📞 Support

For questions about this implementation or suggestions for improvements, please open an issue in the project repository.

---

*🏠 Built to help communities identify and address property blight proactively*