"""
Example Usage of Simple RAG System

This script demonstrates various ways to use the Simple RAG system
for educational and development purposes.
"""

from simple_rag import SimpleRAG


def basic_usage_example():
    """
    Basic example of using the RAG system
    """
    print("🔥 Basic RAG Usage Example")
    print("=" * 50)
    
    # Create RAG system
    rag = SimpleRAG(top_k_retrieval=2)
    
    # Ask a simple question
    question = "What is machine learning?"
    answer = rag.ask(question)
    
    print(f"Q: {question}")
    print(f"A: {answer}")


def batch_processing_example():
    """
    Example of processing multiple questions at once
    """
    print("\n📦 Batch Processing Example")
    print("=" * 50)
    
    rag = SimpleRAG()
    
    questions = [
        "What is Python used for?",
        "How does Flask work?",
        "What are the principles of database design?"
    ]
    
    results = rag.batch_ask(questions)
    
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Q: {result['question']}")
        print(f"   A: {result['answer']}")


def knowledge_base_exploration():
    """
    Example of exploring the knowledge base
    """
    print("\n🔍 Knowledge Base Exploration")
    print("=" * 50)
    
    rag = SimpleRAG()
    
    # Get system information
    info = rag.get_system_info()
    print(f"Available documents: {len(info['available_documents'])}")
    for doc in info['available_documents']:
        print(f"  - {doc}")
    
    # Search without generating answer
    print(f"\nSearching for 'programming':")
    results = rag.search_knowledge_base("programming")
    for item in results:
        doc = item['document']
        score = item['similarity_score']
        print(f"  - '{doc['title']}' (relevance: {score:.3f})")


def detailed_process_example():
    """
    Example showing detailed RAG process
    """
    print("\n🔬 Detailed Process Example")
    print("=" * 50)
    
    rag = SimpleRAG()
    
    question = "How do I learn data science?"
    result = rag.ask_with_explanation(question)
    
    print(f"\nProcess Summary:")
    print(f"  - Documents retrieved: {result['process_details']['num_documents_retrieved']}")
    print(f"  - Context length: {result['process_details']['context_length']} chars")
    print(f"  - Query type: {result['process_details']['query_type']}")
    print(f"  - Strategy: {result['process_details']['generation_strategy']}")


def extending_knowledge_base():
    """
    Example of adding new documents to the knowledge base
    """
    print("\n📚 Extending Knowledge Base")
    print("=" * 50)
    
    rag = SimpleRAG()
    
    # Add a new document
    new_title = "Neural Networks Basics"
    new_content = """
    Neural networks are computing systems inspired by biological neural networks.
    They consist of interconnected nodes (neurons) that process information.
    Deep learning uses multi-layer neural networks to learn complex patterns.
    Popular frameworks include TensorFlow, PyTorch, and Keras.
    Applications include image recognition, natural language processing, and more.
    """
    
    print(f"Adding new document: '{new_title}'")
    rag.add_document(new_title, new_content)
    
    # Retrain embedder for optimal performance
    rag.retrain_embedder()
    
    # Test with the new document
    print(f"\nTesting with new document:")
    answer = rag.ask("What are neural networks?")


def interactive_demo():
    """
    Interactive demo that lets users ask questions
    """
    print("\n💬 Interactive RAG Demo")
    print("=" * 50)
    print("Ask questions about the available topics!")
    print("Available topics: Machine Learning, Python, Data Science, Flask, Databases")
    print("Type 'quit' to exit\n")
    
    rag = SimpleRAG()
    
    while True:
        try:
            question = input("Your question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'bye']:
                print("Goodbye! 👋")
                break
            
            if not question:
                continue
            
            print(f"\n🤖 Processing: '{question}'")
            answer = rag.ask(question)
            print()
            
        except KeyboardInterrupt:
            print("\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"Error: {e}")


def compare_different_queries():
    """
    Compare how the system handles different types of queries
    """
    print("\n⚖️  Query Type Comparison")
    print("=" * 50)
    
    rag = SimpleRAG()
    
    # Different types of queries
    queries = {
        'Definition': "What is machine learning?",
        'How-to': "How do I use Python?",
        'Comparison': "What is the difference between Python and Flask?",
        'General': "Tell me about data science tools"
    }
    
    for query_type, question in queries.items():
        print(f"\n{query_type} Query: '{question}'")
        answer = rag.ask(question)
        print(f"Response length: {len(answer)} characters")


def main():
    """
    Run all examples
    """
    print("🚀 Simple RAG System - Complete Examples")
    print("=" * 80)
    
    # Run all examples
    basic_usage_example()
    batch_processing_example()
    knowledge_base_exploration()
    detailed_process_example()
    extending_knowledge_base()
    compare_different_queries()
    
    # Optional: Run interactive demo
    # interactive_demo()
    
    print("\n✅ All examples completed!")
    print("\nTo run the interactive demo, uncomment the line in main() or run:")
    print("  python -c \"from example_usage import interactive_demo; interactive_demo()\"")


if __name__ == "__main__":
    main()