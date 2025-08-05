#!/usr/bin/env python3
"""
Easy RAG Demo Runner
No-hassle way to run the LangChain RAG example
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages if missing"""
    required_packages = [
        "langchain>=0.1.0",
        "langchain-openai>=0.1.0", 
        "langchain-community>=0.1.0",
        "faiss-cpu>=1.7.0"
    ]
    
    print("📦 Installing required packages...")
    for package in required_packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package], 
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"  ✅ {package}")
        except subprocess.CalledProcessError:
            print(f"  ❌ Failed to install {package}")
            return False
    return True

def run_demo():
    """Run the RAG demo"""
    print("\n🚀 Running RAG Demo...")
    print("=" * 50)
    
    # Set environment variables
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    
    # Import and run
    try:
        from langchain_rag import SimpleRAG
        
        rag = SimpleRAG()
        
        questions = [
            "What is machine learning?",
            "How does web development work?", 
            "Tell me about cybersecurity"
        ]
        
        for question in questions:
            rag.ask(question)
            
        print("\n✅ Demo completed successfully!")
        
    except Exception as e:
        print(f"❌ Error running demo: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("🎯 Easy RAG Demo - No Hassle Setup")
    print("=" * 40)
    
    # Install requirements
    if not install_requirements():
        print("❌ Failed to install requirements")
        sys.exit(1)
    
    # Run demo
    if not run_demo():
        print("❌ Demo failed")
        sys.exit(1)
    
    print("\n🎉 All done! RAG system working perfectly.")