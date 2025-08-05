# MIDAS Hackathon - AI for Social Good Projects

This repository contains machine learning and AI implementations developed for the MIDAS Hackathon, focusing on technical applications for Detroit urban challenges.

## 🚀 Quick Start

### Environment Setup
```bash
# Option 1: Conda (Recommended)
conda env create -f environment.yml
conda activate midas_aug25

# Option 2: Pip
pip install -r requirements.txt
```

NOTE: You may have to install another layer of requirements.txt files within the projects.

### Learning Resources
🎓 **New to ML/RAG?** Start here: [`learning/`](learning/)
- **RAG Tutorial**: `learning/rag_for_proj1/` - Learn LangChain and vector databases
- **Tabular ML Tutorial**: `learning/xgboost_for_proj2/` - Learn XGBoost for classification

**External Resources:**
- **RAG-time**: [Microsoft's RAG cookbook](https://github.com/microsoft/rag-time) - Production RAG patterns
- **XGBoost Guide**: [Complete Guide to Parameter Tuning](https://xgboost.readthedocs.io/en/stable/tutorials/param_tuning.html) - Official XGBoost tuning

## 📊 Projects

### Project 1: Detroit Open Data Portal Enhancement
**Directory**: [`1_detroit_open_data_portal/`](1_detroit_open_data_portal/)

Natural language chatbot for Detroit's Open Data Portal with 200+ city datasets.

**Potential Tech Stack**: LangChain, FAISS, OpenAI API, vector embeddings, RAG (Retrieval-Augmented Generation)

### Project 2: Detroit Computer Vision for Building Habitability
**Directory**: [`2_detroit_computer_vision/`](2_detroit_computer_vision/)

Computer vision tools for building habitability using Detroit imagery spanning 1999-2024.

**Potential Future Components**: Computer vision models for aerial/street imagery analysis, temporal building deterioration analysis

### Project 3: Detroit Flood Risk Policy Analysis
**Directory**: [`3_detroit_flood_risk_analysis/`](3_detroit_flood_risk_analysis/)

Interactive flood/erosion risk policy tool inspired by En-ROADS for stakeholder-driven scenario analysis.

**Potential Tech Stack**: LLM integration for scenario generation, hydrological modeling, geospatial analysis, interactive web platform

## 🛠️ Technical Details

### Core Technologies
- **Machine Learning**: XGBoost, scikit-learn, Optuna (hyperparameter optimization)
- **RAG/LLM**: LangChain, FAISS, OpenAI API, vector embeddings
- **Data Science**: pandas, numpy, matplotlib, seaborn, Jupyter
- **Geospatial**: geopandas, folium, contextily

### Key Features
- **Reproducible environments** with conda/pip specifications
- **Comprehensive evaluation** with balanced metrics for imbalanced data
- **Production-ready code** with proper logging, model persistence
- **Educational examples** in `learning/` directory

## 📁 Repository Structure

```
MIDAS-Hackathon-Aug2025/
├── 1_detroit_open_data_portal/     # RAG for open data search
├── 2_detroit_computer_vision/      # ML for blight classification
│   ├── models/                    # XGBoost implementations
│   ├── training_data/             # Processed datasets
│   ├── deliverables/              # Model outputs & visualizations
│   └── eda/                       # Exploratory data analysis
├── 3_detroit_flood_risk_analysis/  # Policy modeling tool
├── learning/                       # 🎓 Start here for tutorials
│   ├── rag_for_proj1/             # Learn RAG implementation
│   └── xgboost_for_proj2/         # Learn tabular ML
├── data/                          # Raw datasets (not in git)
├── environment.yml                # Conda environment
├── requirements.txt               # Pip requirements
└── SETUP.md                       # Detailed setup guide
```

## 🔬 Data & Performance

### Project 2 - Blight Classification (Baseline Model)
**Current Implementation**: Multi-class blight classification using Detroit Land Bank Authority survey data as foundation for future computer vision work.

**Problem**: Multi-class classification (0=No Blight → 3=Extreme Blight)
**Data**: Detroit Land Bank Authority survey data (~98k property records)
**Features**: Property condition indicators (roof, openings, occupancy, fire damage)
**Class Distribution**: Highly imbalanced (49% class 1, 4% class 3)

**Tech Stack**: XGBoost, scikit-learn, Optuna, pandas, matplotlib

**Baseline Model Results**:
- **XGBoost Baseline**: 62.6% accuracy, 51.4% macro F1
- **XGBoost Optimized**: Bayesian hyperparameter tuning with Optuna
- **Key Finding**: OPENINGS_CONDITION most predictive feature (60% importance)
- **Challenge**: Poor performance on minority classes (severe blight cases)

## 🧪 Running the Code

### Project 2 Models
```bash
cd 2_detroit_computer_vision/models/
python xgboost_baseline.py      # Baseline model
python xgboost_optimized1.py    # Bayesian optimization + advanced features
```

### Learning Tutorials
```bash
# XGBoost Tutorial (Project 2)
cd learning/xgboost_for_proj2/
python generate_synthetic_data.py  # Generate demo data first
python train_blight_model.py       # Learn XGBoost with synthetic data
python predict_blight.py --model models/blight_model_large_dataset.joblib --demo

# RAG Tutorial (Project 1)
cd learning/rag_for_proj1/easy_langchain_rag/
python synthetic_knowledge_base.py  # Generate knowledge base first
python run_rag_demo.py              # Learn RAG implementation
```

## 📈 Technical Contributions

### Machine Learning
- **Bayesian hyperparameter optimization** using Optuna TPE sampler
- **Advanced feature engineering** with interaction terms
- **Proper evaluation** for imbalanced multi-class problems
- **Production pipeline** with model persistence and logging

### RAG Implementation
- **Vector database** setup with FAISS
- **Document chunking** and embedding strategies
- **Graceful degradation** when API keys unavailable

## 🎯 Impact & Applications

**Technical Applications**:
- **Automated property assessment** using tabular data
- **Semantic search** over large document collections
- **Policy scenario modeling** with LLM integration

**Educational Value**:
- **Complete ML pipelines** from data preprocessing to evaluation
- **Best practices** for imbalanced classification
- **RAG implementation** with practical examples

## 🤝 Contributing

Focus areas for technical contributions:
- **Model improvements**: Better handling of class imbalance, ensemble methods
- **Feature engineering**: Time-series features, spatial features from coordinates
- **Evaluation**: Additional metrics, fairness analysis
- **Documentation**: More tutorial examples, advanced techniques

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Feel free to use, modify, and distribute this code for any purpose!

## ⚠️ Disclaimer

This project was developed independently as part of the MIDAS Hackathon. The code, documentation, and all content in this repository represent personal work and opinions, and are not affiliated with, endorsed by, or related to any employer or organization. All views expressed are my own.

---

**🔧 Built with Python, scikit-learn, XGBoost, LangChain, and other modern ML tools**