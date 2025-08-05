"""
Simple LangChain RAG System

Core RAG concepts: Retrieval + Generation + Prompting
Uses OpenAI GPT and FAISS vector store.

Required: pip install langchain langchain-openai langchain-community faiss-cpu
Set environment variable: export KMP_DUPLICATE_LIB_OK=TRUE (for macOS)
"""

import json
import os
from pathlib import Path

# Fix OpenMP library conflict on macOS
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Set a dummy OpenAI API key if not provided (for demo mode)
if not os.getenv('OPENAI_API_KEY'):
    os.environ['OPENAI_API_KEY'] = 'sk-dummy-key-for-demo'
    print("⚠️  No OpenAI API key found. Running in demo mode with mock responses.")
    DEMO_MODE = True
else:
    DEMO_MODE = False

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain_openai import OpenAI, OpenAIEmbeddings
    from langchain.schema import Document
    from langchain.chains import RetrievalQA
    from langchain.prompts import PromptTemplate
    LANGCHAIN_AVAILABLE = True
except ImportError as e:
    print(f"❌ LangChain import error: {e}")
    print("📦 Install missing packages with: pip install langchain langchain-openai langchain-community faiss-cpu")
    LANGCHAIN_AVAILABLE = False


class SimpleRAG:
    """Simple RAG: Retrieval + Generation + Prompting"""
    
    def __init__(self):
        if not LANGCHAIN_AVAILABLE:
            print("❌ Cannot initialize RAG system - LangChain not available")
            self.available = False
            return
            
        try:
            # Load documents and create vector store
            documents = self._load_documents()
            
            if DEMO_MODE:
                print("🔄 Running in demo mode - simulating RAG responses")
                self.available = False
                return
            
            # Set up embeddings and vector store (RETRIEVAL)
            embeddings = OpenAIEmbeddings()
            self.vectorstore = FAISS.from_documents(documents, embeddings)
            
            # Set up LLM (GENERATION)  
            self.llm = OpenAI(temperature=0)
            
            # Create RAG chain with custom prompt (PROMPTING)
            prompt = PromptTemplate(
                template="""Answer the question based on the context below.
                
Context: {context}

Question: {question}

Answer:""",
                input_variables=["context", "question"]
            )
            
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3}),
                chain_type_kwargs={"prompt": prompt},
                return_source_documents=True
            )
            self.available = True
            print("✅ RAG system initialized successfully!")
            
        except Exception as e:
            print(f"❌ Error initializing RAG system: {e}")
            self.available = False
    
    def _load_documents(self):
        """Load and split documents"""
        # Find the knowledge base file
        kb_file = Path("synthetic_knowledge_base.json")
        if not kb_file.exists():
            # Try relative path from script location
            kb_file = Path(__file__).parent / "synthetic_knowledge_base.json"
            
        if not kb_file.exists():
            raise FileNotFoundError("Could not find synthetic_knowledge_base.json")
            
        with open(kb_file, 'r') as f:
            data = json.load(f)
        
        documents = []
        for doc in data['documents']:
            documents.append(Document(
                page_content=f"{doc['title']}\n\n{doc['content']}",
                metadata={"title": doc['title'], "category": doc['category']}
            ))
        
        # Split documents into chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        return splitter.split_documents(documents)
    
    def ask(self, question: str):
        """Ask a question and get an answer with sources"""
        if not self.available:
            # Demo mode responses
            demo_responses = {
                "What is machine learning?": "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed.",
                "How does web development work?": "Web development involves creating websites and web applications using languages like HTML, CSS, and JavaScript, along with backend technologies.",
                "Tell me about cybersecurity": "Cybersecurity involves protecting computer systems, networks, and data from digital attacks, unauthorized access, and other threats."
            }
            
            print(f"\n🤖 Q: {question}")
            response = demo_responses.get(question, f"This is a demo response for: {question}")
            print(f"🤖 A: {response}")
            print("📚 Sources: [Demo mode - no real sources]")
            return {"result": response, "source_documents": []}
        
        try:
            result = self.qa_chain.invoke({"query": question})
            
            print(f"\n✅ Q: {question}")
            print(f"✅ A: {result['result']}")
            print("📚 Sources:")
            for i, doc in enumerate(result['source_documents'], 1):
                print(f"  {i}. {doc.metadata['title']}")
            
            return result
            
        except Exception as e:
            print(f"❌ Error processing question: {e}")
            return {"result": f"Error: {e}", "source_documents": []}


# Demo
if __name__ == "__main__":
    print("🚀 Starting Simple RAG System Demo")
    print("=" * 50)
    
    # Check if we can run
    if not LANGCHAIN_AVAILABLE:
        print("❌ Cannot run demo - missing dependencies")
        print("📦 Install with: pip install langchain langchain-openai langchain-community faiss-cpu")
        exit(1)
    
    # Initialize RAG system
    print("🔄 Initializing RAG system...")
    rag = SimpleRAG()
    
    # Test questions
    questions = [
        "What is machine learning?",
        "How does web development work?", 
        "Tell me about cybersecurity"
    ]
    
    print(f"\n📝 Running {len(questions)} test questions...")
    print("=" * 50)
    
    for question in questions:
        rag.ask(question)
    
    print("\n" + "=" * 50)
    print("✅ Demo completed!")
    
    if DEMO_MODE:
        print("\n💡 To run with real OpenAI API:")
        print("   export OPENAI_API_KEY='your-api-key-here'")
        print("   python langchain_rag.py")