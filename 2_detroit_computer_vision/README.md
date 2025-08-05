# Project 2: Detroit Computer Vision for Building Habitability

## Project Description

Computer vision tools for building habitability using Detroit imagery spanning 1999-2024. This project aims to develop tools to assess building conditions and habitability using visual analysis of Detroit's built environment over time.

**Current Implementation**: Multi-class blight classification using Detroit Land Bank Authority survey data as a foundation for future computer vision work.

**Data Source:** DLBA survey data (20250527_DLBA_survey_data_UM_Detroit.xlsx)
**Problem Type:** Multi-class classification (4 classes)
**Features:** Property condition indicators (roof, openings, occupancy, etc.)
**Target:** Blight severity levels derived from FIELD_DETERMINATION

## Models

### `xgboost_baseline.py`
Baseline XGBoost classifier for 4-class blight prediction with comprehensive evaluation metrics and feature importance analysis.

### `xgboost_optimized1.py`  
Advanced XGBoost model with Bayesian hyperparameter optimization (Optuna), feature engineering, and K-fold cross-validation.

## Quick Start

**Option 1: Use main environment (covers all projects)**
```bash
# From repository root
pip install -r requirements.txt  # or: conda env create -f environment.yml
cd 2_detroit_computer_vision/models/
python xgboost_baseline.py
```

**Option 2: Use model-specific requirements (minimal install)**
```bash
cd 2_detroit_computer_vision/models/
pip install -r requirements-xgboost_baseline.txt
python xgboost_baseline.py

# For optimized model
pip install -r requirements-xgboost_optimized1.txt  
python xgboost_optimized1.py
```

**Outputs**: All results saved to `deliverables/[model_name]/` with visualizations and metrics.

## Directory Structure

```
2_detroit_computer_vision/
├── models/                          # Model scripts and requirements
│   ├── xgboost_baseline.py         # Baseline XGBoost model
│   ├── xgboost_optimized1.py       # Optimized XGBoost model
│   ├── requirements-xgboost_baseline.txt
│   └── requirements-xgboost_optimized1.txt
├── training_data/                   # Processed training datasets
│   ├── blight_features.csv         # Feature matrix
│   └── blight_labels.csv           # Target labels
├── deliverables/                    # Model outputs and artifacts
│   ├── xgboost_baseline/           # Baseline model results
│   └── xgboost_optimized1/         # Optimized model results
└── eda/                            # Exploratory data analysis notebooks
```

## Data

**Target Variable:** FIELD_DETERMINATION mapped to 4 classes:
- 0: No Action (Salvage), NAP (Salvage), Other Resolution Pathways, Vacant (Not Blighted)
- 1: Noticeable Evidence of Blight  
- 2: Significant Evidence of Blight
- 3: Extreme Evidence of Blight

**Features:** Property condition indicators:
- `IS_OCCUPIED` - Whether property is occupied
- `FIRE_DAMAGE_CONDITION` - Fire damage assessment
- `ROOF_CONDITION` - Roof condition rating
- `OPENINGS_CONDITION` - Doors/windows condition
- `IS_OPEN_TO_TRESPASS` - Accessibility to trespassers

**Dataset Size:** ~98,320 property records
**Class Distribution:** 49% class 1, 27% class 0, 20% class 2, 4% class 3

## Usage

### Run Baseline Model
```bash
cd models/
pip install -r requirements-xgboost_baseline.txt
python xgboost_baseline.py
```

### Run Optimized Model  
```bash
cd models/
pip install -r requirements-xgboost_optimized1.txt
python xgboost_optimized1.py
```

## Outputs

Each model generates comprehensive artifacts in `deliverables/[model_name]/`:

**Data Analysis:**
- `data_analytics.json` - Dataset statistics
- `label_distribution.png` - Class distribution plots
- `data_summary.png` - Summary statistics

**Model Performance:**
- `evaluation_results.json` - All metrics and classification report  
- `confusion_matrix.png` - Prediction accuracy visualization
- `metrics_comparison.png` - Performance metrics comparison

**Feature Analysis:**
- `feature_importance.csv` - Feature importance scores
- `feature_importance.png` - Feature importance visualization

**Predictions:**
- `test_predictions.csv` - Test set predictions with PARCEL_IDs and probabilities

**Additional (Optimized Model):**
- `optimization_results.json` - Hyperparameter optimization results
- `cv_results.json` - Cross-validation results with confidence intervals
- `best_model.pkl` - Serialized trained model

## Requirements

**Core Dependencies:**
- pandas >= 2.0.0
- numpy >= 1.24.0  
- scikit-learn >= 1.3.0
- xgboost >= 2.0.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0

**Additional (Optimized Model):**
- optuna >= 3.4.0
- joblib >= 1.3.0

## Performance

**Baseline Model Results:**
- Accuracy: ~62.6%
- Balanced Accuracy: ~49.7% 
- Macro F1: ~51.4%
- Weighted F1: ~60.5%

**Key Findings:**
- Model performs well on classes 0 and 1 (F1 ~0.66-0.70)
- Struggles with minority classes 2 and 3 (F1 ~0.35)
- OPENINGS_CONDITION is the most important feature (60% importance)
- ROOF_CONDITION and IS_OCCUPIED are secondary features

The models handle class imbalance through balanced evaluation metrics, stratified sampling, and proper train/validation/test splits.

## Future Development Recommendations

### Recommendation #1: VLM + Tabular Hybrid

Take urbanworm and modify in these ways:

1. Split a "single parcel classification" of opening, roof, facade. This becomes 3 VLM classifications.
   - Few-shot prompt each of these:
     - Show examples of opening exposed
     - Show roof being damaged/broken

2. Add the three columns (opening, roof, facade) to the most predictive columns from tabular data like water_status or usps_vacancy.

**Modeling recommendations:**
- Try Qwen-2.5-VL-7B-instruct-AWQ (for a 4090)
- Can also use phi-4-multimodal (6B params)
  - https://huggingface.co/microsoft/Phi-4-multimodal-instruct

Recommend trying to get Azure credits to use GPT-4o-mini or GPT-4.1-mini through API.
- Will enable easy versioning of prompts
- Definitely will cost

### Recommendation #2: Tabular data approach, but better

1. Clean and de-dupe the data better, this will give a better baseline to work with.

2. Add in neighborhood or area level data to parcels:
   - In this radius, this is how many fires in the last 1 year...
   - Assessed value of surrounding 100 homes.

At the end of this, can try MANY tabular models:
- xgboost
- lightgbm
- autogluon

#### Technical Risks (tabular ML data approach)
- Time series data, difficult to predict against without making time series models.
  - Need to do filtering or discretization (i.e. <1yr, <2yr)

- **#1 Duplication of data** based on surveys, permits issued, usps vacancy checks, water status checks.

- **#2 Deduplication of addresses/units** per parcel ID
  - You have to consider the determination per unit vs. per parcel, they might be very different.

- **Data distribution shift of labels.**
  - The blight survey only does vacant homes, but we want to predict against all of them.
  - This means that we will have a major shift between what we've trained on.

### Recommendation #3: Infinite resources

Feed in every piece of data:
- All tabular data columns, per parcel
- Street view images, per parcel
- Top-down satellite images, per parcel

Make a very elaborate prompt, few-shot:

```
You are an expert housing blight detector. Here are the 4 categories of Blight.
0: No Blight
1: Noticeable blight
2: Significant Blight
3: Extreme blight

Here is an example of no blight (INSERT IMAGES AND COLUMNS of EXAMPLE NO BLIGHT HOUSE)

Here is an example of noticeable blight (INSERT IMAGES AND COLUMNS of EXAMPLE NOTICEABLE BLIGHT HOUSE)

...

Please predict this parcel for blight given all the context.

(INSERT ONE HOME OF IMAGES AND TABULAR DATA, "unrolled")
```

**Example costing for GPT-4.1**
- ESTIMATED: 20k parcels, 30k input tokens, 2k output tokens.

Model pricing:
• Prompt (input) tokens: $0.06 per 1,000 tokens
• Completion (output) tokens: $0.12 per 1,000 tokens 
Microsoft Azure

Total tokens:
• Prompt: 30,000 tokens × 20,000 samples = 600,000,000 tokens
• Completion: 2,000 tokens × 20,000 samples = 40,000,000 tokens

Cost:
• Prompt cost: (600,000,000 / 1,000) × $0.06 = $36,000
• Completion cost: (40,000,000 / 1,000) × $0.12 = $4,800

**Total estimate: $40,800**

### Current Computer Vision Potential

**Planned Features**:
- Aerial imagery analysis for building condition assessment
- Street-level imagery processing for habitability metrics
- Temporal analysis of building deterioration (1999-2024)
- Integration of survey data with visual analysis

**Tech Stack**: OpenCV, PyTorch/TensorFlow, VLMs, satellite/aerial imagery APIs