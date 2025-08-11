# Project 1: Detroit Open Data Portal Enhancement

## Project Description

Natural language chatbot for Detroit's Open Data Portal with 200+ city datasets. The project aims to develop a chatbot and search enhancement system that improves user experience within Detroit's Open Data Portal, which houses over 200 datasets spanning multiple city departments.

**Key Goals**:
- Enable natural language querying of municipal datasets
- Improve dataset discoverability across city departments
- Bridge the gap between technical data and practical citizen needs
- Support both resident service needs and researcher analysis

## Challenges

Users currently struggle with:
- Finding relevant datasets among 200+ available options
- Understanding dataset relationships and metadata
- Accessing information in natural, intuitive ways
- Navigating complex municipal data structures

## Technical Approach

**Tech Stack**: LangChain, ChromaDB, Ollama, vector embeddings [Nomic embedding text], RAG (Retrieval-Augmented Generation)

## 🚀 Features
### Core Capabilities

- **📊 Interactive Dashboards** - Real-time visualizations of incident data and performance metrics
- **💬 Natural Language Queries** - Ask questions about your data in plain English
- **🗺️ Geographic Analysis** - Interactive maps showing incident locations and patterns
- **📈 Advanced Analytics** - Statistical analysis with filtering and trend identification
- **📁 Data Management** - Support for GeoJSON data import and processing
- **⏱️ Performance Monitoring** - Response time analysis and operational insights

## 🚀 Quick Start

### Prerequisites

1. **Python 3.8+**
2. **Ollama** - Install from [ollama.ai](https://ollama.ai)
3. **Required Ollama Models**:
   ```bash
   ollama pull llama3.2:latest
   ollama pull nomic-embed-text:latest
   ```

### Installation
1. **Clone the repository**:
   ```bash
   git clone https://github.com/Chudbrochil/MIDAS-Hackathon-Aug2025.git
   cd 1_detroit_open_data_portal
   git checkout trunk
   ```
2. **Create a virtual environment and install dependencies**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
   
3. **Prepare your data**:
   - Place your data JSON file in the `data/` directory [similar to sample.json file]
   - Update `default_data_path` in `app.py` if needed 

4. **Run the application**:
   ```bash
   streamlit run app.py
   ```

5. **Initialize the system**:
   - Click "Load sample data" from the main screen **If you dont have data available**
   - Wait for data loading and embedding (first-time setup takes longer)

The application will open automatically in your default web browser

## Upload Your Own Data

- Go to the Data Upload page from the side navbar
- Upload a GeoJSON file containing incident data
- Preview and load your data into the system

## 📁Project Structure
```
detroit-open-data-portal/
├── data/
│    └── sample.json
├── src/
│   ├── data_models.py      # Data structures and validation
│   ├── data_manager.py     # Data processing and storage
│   ├── query_processor.py  # Natural language processing 
│   └── utils.py            # Utility functions
├── app.py                  # Streamlit application
├── requirements.txt        # Dependencies
└── README.md               # This file
```
## 📋 Requirements

### Python Dependencies
```
streamlit
pandas
numpy
pydantic
langchain
langchain-community
langchain-ollama
langchain-chroma
chromadb
requests
plotly
geopy
Pillow
python-dateutil
python-dotenv
chromadb
ollama
requests
beautifulsoup4
```
### System Requirements
- **RAM**: 8GB minimum (16GB recommended for large datasets)
- **Storage**: 2GB+ for embeddings and cache
- **CPU**: Multi-core recommended for embedding generation
- **GPU**: Optional (can accelerate Ollama models)
- **OS**: Windows, macOS, or Linux with Docker support

### Ollama Requirements
- **Models**: granite3-dense:8b (~5GB), nomic-embed-text (~274MB)
- **VRAM**: 6GB+ for GPU acceleration (optional)
- **CPU**: 4+ cores recommended for good performance

## Learning Resources

🎓 **New to RAG?** Start here: [`../learning/rag_for_proj1/`](../learning/rag_for_proj1/)

Complete tutorial on LangChain and vector databases with practical examples.

**Note**: This is a research and analytics tool designed for city of detroit open portal data. Ensure compliance with data privacy policies and data governance requirements when working with sensitive data. It can be adapted for other data with similar GeoJSON structures.
