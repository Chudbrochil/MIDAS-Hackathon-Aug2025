# Environment Setup Guide

## Quick Start with Conda (Recommended)

### 1. Create and activate the environment:
```bash
# Create environment from file
conda env create -f environment.yml

# Activate environment
conda activate midas_aug25
```

### 2. Verify installation:
```bash
python -c "import pandas, numpy, sklearn, xgboost, jupyter; print('✅ All core libraries installed!')"
```

## Alternative: Using pip only

If you prefer pip or don't have conda:

```bash
# Create virtual environment
python -m venv midas_aug25
source midas_aug25/bin/activate  # On Windows: midas_aug25\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

## Environment Management Best Practices

### ✅ **DO** (Recommended Approach):
- **Use `environment.yml`** - This is the best practice for conda environments
- **Version control the environment file** - Others can recreate exact environment
- **Use conda-forge channel** - Better package compatibility
- **Pin major versions** - Ensures reproducibility while allowing updates

### 🚫 **DON'T** (Bad Practices):
- Don't commit the actual environment folder to git
- Don't use `conda env export > environment.yml` (includes too many dependencies)
- Don't mix conda and pip unless necessary

## What's Included

### 🔬 **Data Science Stack:**
- pandas, numpy, matplotlib, seaborn
- jupyter, jupyterlab, ipykernel

### 🤖 **Machine Learning:**
- scikit-learn, xgboost, joblib

### 🗂️ **Data Handling:**
- openpyxl (Excel files), geopandas (GIS data)

### 🤖 **AI/RAG Projects:**
- langchain, langchain-openai, faiss-cpu

### 🗺️ **Geospatial (Detroit data):**
- geopandas, folium, contextily

## Project Structure Support

This environment supports all projects in the repository:

- **Project 1**: RAG chatbots (langchain dependencies)
- **Project 2**: Blight classification (ML/GIS dependencies) 
- **Project 3**: Flood risk analysis (geospatial dependencies)
- **EDA**: All Jupyter notebooks and data analysis

## Updating the Environment

When you add new dependencies:

```bash
# Add to environment.yml, then update
conda env update -f environment.yml --prune

# Or for pip users
pip install new_package
pip freeze > requirements.txt  # Only if needed
```

## Troubleshooting

### Common Issues:

1. **Import errors**: Make sure environment is activated
2. **Jupyter kernel issues**: Install ipykernel in the environment
3. **Excel file errors**: Install openpyxl
4. **GIS data issues**: Install geopandas

### Getting Help:
```bash
# Check what's installed
conda list

# Check environment
conda info --envs

# Test core functionality
python -c "import pandas as pd; print(f'Pandas version: {pd.__version__}')"
```