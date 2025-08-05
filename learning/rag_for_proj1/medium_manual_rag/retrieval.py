"""
Document Retrieval Module for RAG System

This module handles the "Retrieval" part of RAG (Retrieval-Augmented Generation).
It takes a user query and finds the most relevant documents from the knowledge base
using semantic similarity based on embeddings.

The retrieval process:
1. Embed the user query using the same embedding model as documents
2. Calculate similarity between query embedding and all document embeddings
3. Rank documents by similarity score
4. Return top-k most relevant documents
"""

from typing import List, Dict, Tuple
from embeddings import SimpleEmbedder
from fake_database import FakeDatabase


class DocumentRetriever:
    """
    Retrieves relevant documents based on query similarity
    
    This class implements semantic search by comparing query embeddings
    to document embeddings and returning the most similar documents.
    """
    
    def __init__(self, database: FakeDatabase, embedder: SimpleEmbedder):
        """
        Initialize the retriever with database and embedder
        
        Args:
            database (FakeDatabase): Database containing documents
            embedder (SimpleEmbedder): Trained embedding model
        """
        self.database = database
        self.embedder = embedder
        
        # Ensure embedder is trained on the document corpus
        if not self.embedder.is_fitted:
            print("Training embedder on document corpus...")
            documents = self.database.get_all_documents()
            self.embedder.fit(documents)
    
    def retrieve_documents(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Retrieve top-k most relevant documents for a given query
        
        Args:
            query (str): User query/question
            top_k (int): Number of documents to retrieve (default: 3)
            
        Returns:
            List[Dict]: List of most relevant documents with similarity scores
        """
        print(f"Retrieving top {top_k} documents for query: '{query}'")
        
        # Step 1: Embed the user query
        query_vector = self.embedder.embed_text(query)
        
        # Step 2: Calculate similarity with all documents
        document_scores = []
        all_documents = self.database.get_all_documents()
        
        for doc in all_documents:
            # Get pre-computed document vector
            doc_vector = self.embedder.get_document_vector(doc['id'])
            
            # Calculate similarity between query and document
            similarity = self.embedder.calculate_similarity(query_vector, doc_vector)
            
            # Store document with its similarity score
            document_scores.append({
                'document': doc,
                'similarity_score': similarity
            })
        
        # Step 3: Sort documents by similarity score (highest first)
        document_scores.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        # Step 4: Return top-k documents
        top_documents = document_scores[:top_k]
        
        # Log retrieval results for transparency
        print("Retrieval Results:")
        for i, item in enumerate(top_documents, 1):
            doc = item['document']
            score = item['similarity_score']
            print(f"  {i}. '{doc['title']}' (similarity: {score:.3f})")
        
        return top_documents
    
    def retrieve_with_threshold(self, query: str, similarity_threshold: float = 0.1) -> List[Dict]:
        """
        Retrieve documents above a certain similarity threshold
        
        This method is useful when you want to ensure minimum relevance
        rather than a fixed number of documents.
        
        Args:
            query (str): User query/question
            similarity_threshold (float): Minimum similarity score (0-1)
            
        Returns:
            List[Dict]: List of relevant documents above threshold
        """
        print(f"Retrieving documents with similarity > {similarity_threshold} for query: '{query}'")
        
        # Get all documents with scores
        all_results = self.retrieve_documents(query, top_k=len(self.database.get_all_documents()))
        
        # Filter by threshold
        relevant_documents = [
            result for result in all_results 
            if result['similarity_score'] > similarity_threshold
        ]
        
        print(f"Found {len(relevant_documents)} documents above threshold")
        return relevant_documents
    
    def get_context_from_documents(self, retrieved_docs: List[Dict], max_length: int = 1000) -> str:
        """
        Combine retrieved documents into a single context string
        
        This method takes the retrieved documents and formats them into
        a single string that can be used as context for generation.
        
        Args:
            retrieved_docs (List[Dict]): Documents returned by retrieve_documents
            max_length (int): Maximum length of combined context
            
        Returns:
            str: Formatted context string
        """
        if not retrieved_docs:
            return "No relevant documents found."
        
        context_parts = []
        current_length = 0
        
        for item in retrieved_docs:
            doc = item['document']
            score = item['similarity_score']
            
            # Format document with title and content
            doc_text = f"Title: {doc['title']}\nContent: {doc['content'].strip()}\n"
            
            # Check if adding this document would exceed max_length
            if current_length + len(doc_text) > max_length and context_parts:
                break
            
            context_parts.append(doc_text)
            current_length += len(doc_text)
        
        # Combine all document parts
        context = "\n" + "="*50 + "\n".join(context_parts)
        
        return context
    
    def explain_retrieval(self, query: str, top_k: int = 3) -> None:
        """
        Detailed explanation of the retrieval process for educational purposes
        
        Args:
            query (str): User query to analyze
            top_k (int): Number of documents to retrieve
        """
        print(f"\n{'='*60}")
        print("RETRIEVAL PROCESS EXPLANATION")
        print(f"{'='*60}")
        
        print(f"\n1. QUERY ANALYSIS:")
        print(f"   Original query: '{query}'")
        
        # Show query preprocessing
        query_words = self.embedder._preprocess_text(query)
        print(f"   Processed words: {query_words}")
        
        # Show query embedding
        query_vector = self.embedder.embed_text(query)
        top_query_terms = sorted(query_vector.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"   Top query terms (TF-IDF): {top_query_terms}")
        
        print(f"\n2. DOCUMENT COMPARISON:")
        all_documents = self.database.get_all_documents()
        
        for doc in all_documents:
            doc_vector = self.embedder.get_document_vector(doc['id'])
            similarity = self.embedder.calculate_similarity(query_vector, doc_vector)
            
            print(f"   '{doc['title']}': {similarity:.3f} similarity")
            
            # Show overlapping terms
            common_terms = set(query_vector.keys()) & set(doc_vector.keys())
            if common_terms:
                print(f"     Common terms: {list(common_terms)[:5]}")
        
        print(f"\n3. FINAL RANKING:")
        results = self.retrieve_documents(query, top_k)
        for i, item in enumerate(results, 1):
            doc = item['document']
            score = item['similarity_score']
            print(f"   Rank {i}: '{doc['title']}' (score: {score:.3f})")
        
        print(f"{'='*60}\n")