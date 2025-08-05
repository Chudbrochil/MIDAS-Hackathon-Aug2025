# Simple RAG System

A simple, educational implementation of a **Retrieval-Augmented Generation (RAG)** system built from scratch in Python. This project demonstrates the core concepts of RAG without external dependencies on large language models or vector databases.

## 🎯 Purpose

This RAG system is designed as a **template and learning tool** for university projects and educational purposes. It provides a clear, well-commented foundation that can be extended into more sophisticated systems.

## 🏗️ Architecture

The system consists of five main components:

```
Query → Embedding → Retrieval → Generation → Response
```

### Components

1. **Database** (`fake_database.py`)
   - In-memory document storage with sample data
   - Basic CRUD operations and keyword search
   - Contains sample documents on programming topics

2. **Embeddings** (`embeddings.py`)
   - TF-IDF based text embedding (implemented from scratch)
   - Document vectorization and similarity calculation
   - No external dependencies required

3. **Retrieval** (`retrieval.py`)
   - Semantic document retrieval using cosine similarity
   - Configurable number of documents to retrieve
   - Context preparation for generation

4. **Generation** (`generation.py`)
   - Template-based response generation
   - Query classification (definition, how-to, comparison, general)
   - Source attribution and context integration

5. **RAG Pipeline** (`simple_rag.py`)
   - Main orchestrator that combines all components
   - Simple API for question-answering
   - Detailed process explanation for educational purposes

## 🚀 Quick Start

### Basic Usage

```python
from simple_rag import SimpleRAG

# Initialize the RAG system
rag = SimpleRAG()

# Ask a question
answer = rag.ask("What is machine learning?")
print(answer)
```

### Running the Demo

```bash
# Run the main demo
python simple_rag.py

# Run comprehensive examples
python example_usage.py
```

## 📖 Example Interactions

### Question: "What is machine learning?"
**Answer:** Based on the retrieved information, machine learning is a subset of artificial intelligence that enables computers to learn and make decisions without being explicitly programmed. Common types include supervised learning, unsupervised learning, and reinforcement learning.

**Sources:** 'Introduction to Machine Learning' (relevance: 0.42)

### Question: "How do I use Python?"
**Answer:** To use Python, you should start with its high-level, interpreted programming language known for its simplicity and readability. Python is widely used in web development, data science, artificial intelligence, and automation.

## 🔧 Key Features

- **Zero External Dependencies**: Built using only Python standard library
- **Educational Focus**: Extensive comments and explanations
- **Modular Design**: Each component can be understood and modified independently
- **Process Transparency**: Detailed explanations of retrieval and generation steps
- **Extensible**: Easy to replace components with more sophisticated alternatives

## 📊 System Information

- **Documents**: 5 sample documents covering programming topics
- **Embedding Method**: TF-IDF with custom implementation
- **Retrieval**: Cosine similarity-based ranking
- **Generation**: Template-based with query classification
- **Vocabulary**: ~156 unique terms after preprocessing

## 🎓 Educational Use

This system is perfect for:

- **Understanding RAG concepts** without getting lost in complex implementations
- **Learning information retrieval** principles
- **Exploring text processing** and similarity calculations
- **Building more advanced systems** as a starting foundation
- **Research projects** requiring a simple baseline

## 🔄 Extending the System

### Replace Components

1. **Database**: Connect to real databases, APIs, or document stores
2. **Embeddings**: Use sentence-transformers, OpenAI embeddings, or other models
3. **Retrieval**: Implement vector databases like Pinecone, Weaviate, or Chroma
4. **Generation**: Integrate LLMs like GPT, Claude, or local models

### Example Extensions

```python
# Add new documents
rag.add_document("New Topic", "Content about the new topic...")
rag.retrain_embedder()  # Update embeddings

# Batch processing
questions = ["Question 1?", "Question 2?"]
results = rag.batch_ask(questions)

# Detailed analysis
result = rag.ask_with_explanation("Your question?")
```

## 📝 Files Overview

- `fake_database.py` - Sample document storage and basic search
- `embeddings.py` - TF-IDF embedding implementation
- `retrieval.py` - Document retrieval and ranking
- `generation.py` - Response generation with templates
- `simple_rag.py` - Main RAG system orchestrator
- `example_usage.py` - Comprehensive usage examples
- `README.md` - This documentation

## 🎯 Learning Outcomes

After studying this implementation, you'll understand:

- How RAG systems work end-to-end
- Text preprocessing and embedding techniques
- Information retrieval and ranking
- Template-based text generation
- System architecture and component integration

## 🚧 Limitations

This is a **simplified educational implementation**:

- Basic TF-IDF embeddings (no semantic understanding)
- Template-based generation (no actual language model)
- Small document collection (5 sample documents)
- No persistent storage (in-memory only)

For production use, consider upgrading to:
- Transformer-based embeddings
- Large language models for generation
- Vector databases for scale
- Persistent storage solutions

## 🎉 Next Steps

1. **Run the examples** to see the system in action
2. **Study the code** to understand each component
3. **Modify components** to experiment with different approaches
4. **Extend the system** with real databases and models
5. **Build your own** advanced RAG system using this as a foundation

This implementation provides a solid foundation for understanding and building more sophisticated RAG systems!