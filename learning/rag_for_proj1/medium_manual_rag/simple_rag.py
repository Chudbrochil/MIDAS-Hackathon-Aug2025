"""
Simple RAG (Retrieval-Augmented Generation) System

This is the main module that combines all components to create a complete RAG system:
1. Database with sample documents (fake_database.py)
2. Text embedding using TF-IDF (embeddings.py)
3. Document retrieval based on similarity (retrieval.py)
4. Response generation using templates (generation.py)

RAG Pipeline:
Query → Embed Query → Retrieve Similar Documents → Generate Response using Context

This implementation is designed for educational purposes and provides a foundation
that can be extended with more sophisticated components like:
- Real vector databases (Pinecone, Weaviate, Chroma)
- Transformer-based embeddings (sentence-transformers, OpenAI)
- Large language models (GPT, Claude, Llama)
"""

from fake_database import FakeDatabase
from embeddings import SimpleEmbedder
from retrieval import DocumentRetriever
from generation import SimpleGenerator
from typing import Dict, List, Optional


class SimpleRAG:
    """
    Complete RAG system that orchestrates all components
    
    This class provides a simple interface for question-answering
    using retrieval-augmented generation.
    """
    
    def __init__(self, top_k_retrieval: int = 3):
        """
        Initialize the RAG system with all components
        
        Args:
            top_k_retrieval (int): Number of documents to retrieve for each query
        """
        print("Initializing Simple RAG System...")
        
        # Initialize all components
        self.database = FakeDatabase()
        self.embedder = SimpleEmbedder()
        self.retriever = DocumentRetriever(self.database, self.embedder)
        self.generator = SimpleGenerator()
        
        # Configuration
        self.top_k = top_k_retrieval
        
        print(f"RAG System initialized with {self.database.get_document_count()} documents")
        print(f"Retrieval set to top-{self.top_k} documents\n")
    
    def ask(self, question: str) -> str:
        """
        Main interface for asking questions to the RAG system
        
        Args:
            question (str): User question
            
        Returns:
            str: Generated answer based on retrieved context
        """
        print(f"{'='*80}")
        print(f"QUESTION: {question}")
        print(f"{'='*80}")
        
        # Step 1: Retrieve relevant documents
        retrieved_docs = self.retriever.retrieve_documents(question, self.top_k)
        
        # Step 2: Format context from retrieved documents
        context = self.retriever.get_context_from_documents(retrieved_docs)
        
        # Step 3: Generate response using context
        answer = self.generator.generate_response(question, context, retrieved_docs)
        
        print(f"\nANSWER:")
        print(f"{answer}")
        print(f"{'='*80}\n")
        
        return answer
    
    def ask_with_explanation(self, question: str) -> Dict[str, any]:
        """
        Ask a question with detailed explanation of the RAG process
        
        This method is useful for understanding how the system works
        and for debugging/educational purposes.
        
        Args:
            question (str): User question
            
        Returns:
            Dict: Complete breakdown of the RAG process
        """
        print(f"{'='*80}")
        print(f"RAG SYSTEM - DETAILED PROCESS")
        print(f"QUESTION: {question}")
        print(f"{'='*80}")
        
        # Step 1: Document Retrieval (with explanation)
        print("\n🔍 STEP 1: DOCUMENT RETRIEVAL")
        print("-" * 40)
        self.retriever.explain_retrieval(question, self.top_k)
        retrieved_docs = self.retriever.retrieve_documents(question, self.top_k)
        
        # Step 2: Context Preparation
        print("\n📄 STEP 2: CONTEXT PREPARATION")
        print("-" * 40)
        context = self.retriever.get_context_from_documents(retrieved_docs)
        print(f"Context length: {len(context)} characters")
        print(f"Context preview: {context[:200]}...")
        
        # Step 3: Response Generation (with explanation)
        print("\n🤖 STEP 3: RESPONSE GENERATION")
        print("-" * 40)
        generation_result = self.generator.generate_with_explanation(question, context, retrieved_docs)
        
        # Compile results
        result = {
            'question': question,
            'retrieved_documents': retrieved_docs,
            'context': context,
            'answer': generation_result['response'],
            'process_details': {
                'num_documents_retrieved': len(retrieved_docs),
                'context_length': len(context),
                'query_type': generation_result['query_type'],
                'generation_strategy': generation_result['generation_strategy']
            }
        }
        
        return result
    
    def batch_ask(self, questions: List[str]) -> List[Dict[str, str]]:
        """
        Process multiple questions in batch
        
        Args:
            questions (List[str]): List of questions to process
            
        Returns:
            List[Dict[str, str]]: List of question-answer pairs
        """
        print(f"Processing {len(questions)} questions in batch...\n")
        
        results = []
        for i, question in enumerate(questions, 1):
            print(f"Processing question {i}/{len(questions)}")
            answer = self.ask(question)
            results.append({
                'question': question,
                'answer': answer
            })
        
        return results
    
    def get_system_info(self) -> Dict[str, any]:
        """
        Get information about the RAG system configuration
        
        Returns:
            Dict: System information and statistics
        """
        return {
            'total_documents': self.database.get_document_count(),
            'vocabulary_size': len(self.embedder.vocabulary) if self.embedder.is_fitted else 0,
            'top_k_retrieval': self.top_k,
            'embedder_type': 'TF-IDF',
            'generator_type': 'Template-based',
            'available_documents': [doc['title'] for doc in self.database.get_all_documents()]
        }
    
    def search_knowledge_base(self, query: str) -> List[Dict]:
        """
        Search the knowledge base without generating an answer
        
        Useful for exploring what documents are available for a topic.
        
        Args:
            query (str): Search query
            
        Returns:
            List[Dict]: Retrieved documents with similarity scores
        """
        print(f"Searching knowledge base for: '{query}'")
        return self.retriever.retrieve_documents(query, self.top_k)
    
    def add_document(self, title: str, content: str) -> None:
        """
        Add a new document to the knowledge base
        
        Note: After adding documents, you should retrain the embedder
        for optimal performance.
        
        Args:
            title (str): Document title
            content (str): Document content
        """
        # Find next available ID
        existing_ids = [doc['id'] for doc in self.database.get_all_documents()]
        new_id = max(existing_ids) + 1 if existing_ids else 1
        
        # Add to database
        new_doc = {
            'id': new_id,
            'title': title,
            'content': content
        }
        
        self.database.documents[new_id] = new_doc
        
        print(f"Added document: '{title}' (ID: {new_id})")
        print("Note: Consider retraining embedder for optimal performance")
    
    def retrain_embedder(self) -> None:
        """
        Retrain the embedder on all documents in the database
        
        Call this after adding new documents to ensure optimal retrieval.
        """
        print("Retraining embedder on updated document corpus...")
        documents = self.database.get_all_documents()
        self.embedder.fit(documents)
        print("Embedder retrained successfully!")


def demo_rag_system():
    """
    Demonstration of the RAG system with example queries
    
    This function shows how to use the RAG system and demonstrates
    its capabilities with various types of questions.
    """
    print("🚀 Simple RAG System Demo")
    print("=" * 50)
    
    # Initialize RAG system
    rag = SimpleRAG(top_k_retrieval=2)
    
    # Display system info
    info = rag.get_system_info()
    print(f"System Info:")
    print(f"  - Documents: {info['total_documents']}")
    print(f"  - Vocabulary: {info['vocabulary_size']} words")
    print(f"  - Available topics: {', '.join(info['available_documents'])}")
    
    # Example queries
    example_questions = [
        "What is machine learning?",
        "How do I use Python?",
        "Tell me about data science",
        "What is Flask used for?",
        "How do databases work?"
    ]
    
    print(f"\n📝 Example Questions:")
    for question in example_questions:
        answer = rag.ask(question)
    
    # Demonstration with detailed explanation
    print(f"\n🔍 Detailed Process Example:")
    result = rag.ask_with_explanation("What is the difference between machine learning and data science?")
    
    return rag


if __name__ == "__main__":
    # Run the demo when script is executed directly
    demo_rag_system()