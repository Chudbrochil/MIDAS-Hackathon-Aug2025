# Simple LangChain RAG System

A minimal RAG implementation focusing on the **core concepts**: Retrieval + Generation + Prompting.

## 🎯 Core RAG Concepts

1. **RETRIEVAL**: Vector search to find relevant documents
2. **GENERATION**: LLM to create answers  
3. **PROMPTING**: Template to combine context + question

## 🚀 No-Hassle Quick Start

### Option 1: Super Easy (Recommended)
```bash
# Just run this - it handles everything
python run_rag_demo.py
```

### Option 2: Manual Setup
```bash
# Install from main requirements (from repo root)
pip install -r ../../../requirements.txt

# OR install just RAG dependencies
pip install langchain langchain-openai langchain-community faiss-cpu

# Run the demo (works without API key in demo mode)
python langchain_rag.py
```

### Option 3: With Real OpenAI API
```bash
# Set your API key for real responses
export OPENAI_API_KEY="your-api-key"
python langchain_rag.py
```

## ✅ What's Fixed

- **No more crashes**: Handles OpenMP library conflicts automatically
- **No API key required**: Runs in demo mode with mock responses
- **Better error handling**: Clear messages if dependencies missing
- **Auto-setup script**: `run_rag_demo.py` installs everything for you

## 🔧 Troubleshooting

**macOS OpenMP Error?** → Fixed automatically (sets `KMP_DUPLICATE_LIB_OK=TRUE`)
**Missing dependencies?** → Run `python run_rag_demo.py` 
**No API key?** → That's fine! Demo mode works without it

# Generate knowledge base  
python synthetic_knowledge_base.py

# Run RAG system
python langchain_rag.py
```

## 💻 Code Overview

The entire RAG system in ~30 lines:

```python
class SimpleRAG:
    def __init__(self):
        # RETRIEVAL: Vector store with embeddings
        documents = self._load_documents()
        embeddings = OpenAIEmbeddings()
        self.vectorstore = FAISS.from_documents(documents, embeddings)
        
        # GENERATION: OpenAI LLM
        self.llm = OpenAI(temperature=0)
        
        # PROMPTING: Custom template  
        prompt = PromptTemplate(template="Answer based on context...")
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.vectorstore.as_retriever(),
            chain_type_kwargs={"prompt": prompt}
        )
    
    def ask(self, question):
        return self.qa_chain({"query": question})
```

## 💻 Usage Examples

```python
rag = SimpleRAG()
result = rag.ask("What is machine learning?")
```

Output:
```
Q: What is machine learning?
A: Machine learning is a subset of artificial intelligence that enables computers to learn...
Sources:
  1. Machine Learning Fundamentals and Applications
  2. Artificial Intelligence and Neural Networks
```

## 🔧 Customization

### Different LLMs
```python
# Anthropic Claude
from langchain_anthropic import ChatAnthropic
self.llm = ChatAnthropic()

# Local model
from langchain_community.llms import Ollama  
self.llm = Ollama(model="llama2")
```

### Custom Prompts
```python
prompt = PromptTemplate(
    template="""You are a helpful assistant. Context: {context}
    Question: {question}
    Answer concisely:""",
    input_variables=["context", "question"]
)
```

## ⚖️ vs Manual RAG

| Aspect | Manual RAG | LangChain RAG |
|--------|------------|---------------|
| **Code** | ~800 lines | ~30 lines |
| **Setup** | None | `pip install` |
| **Flexibility** | Full control | Framework patterns |
| **Learning** | Deep concepts | Quick results |

## 🎯 Key Takeaways

- **LangChain abstracts complexity** - Focus on business logic, not infrastructure
- **RAG = Retrieval + Generation + Prompting** - Three core concepts to master  
- **Vector search enables semantic retrieval** - Better than keyword matching
- **Prompt engineering drives quality** - Template design matters

Start with manual RAG to understand fundamentals, then use LangChain for production speed.