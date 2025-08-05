# Project 1: Detroit Open Data Portal Enhancement

## Project Description

Natural language chatbot for Detroit's Open Data Portal with 200+ city datasets. The project aims to develop a chatbot and search enhancement system that improves user experience within Detroit's Open Data Portal, which houses over 200 datasets spanning multiple city departments.

**Key Goals**:
- Enable natural language querying of municipal datasets
- Improve dataset discoverability across city departments
- Bridge the gap between technical data and practical citizen needs
- Support both resident service needs and researcher analysis

## Current Challenges

Users currently struggle with:
- Finding relevant datasets among 200+ available options
- Understanding dataset relationships and metadata
- Accessing information in natural, intuitive ways
- Navigating complex municipal data structures

## Potential Technical Approach

**Potential Tech Stack**: LangChain, FAISS, OpenAI API, vector embeddings, RAG (Retrieval-Augmented Generation)

**Planned Features**:
- Natural language search across municipal datasets
- Semantic understanding of user queries
- Vector database for efficient document retrieval
- Enhanced metadata mapping and categorization

## Learning Resources

🎓 **New to RAG?** Start here: [`../learning/rag_for_proj1/`](../learning/rag_for_proj1/)

Complete tutorial on LangChain and vector databases with practical examples.

### Quick Demo
```bash
cd ../learning/rag_for_proj1/easy_langchain_rag/
python synthetic_knowledge_base.py  # Generate knowledge base
python run_rag_demo.py              # Run RAG demo
```