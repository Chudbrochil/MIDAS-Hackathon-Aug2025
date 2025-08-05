"""
Simple Text Embedding Module for RAG System

This module provides basic text embedding functionality using simple techniques.
In a production RAG system, you would typically use pre-trained models like
sentence-transformers, OpenAI embeddings, or other transformer-based embeddings.

For simplicity, we're using TF-IDF (Term Frequency-Inverse Document Frequency)
which is implemented from scratch to avoid external dependencies.
"""

import math
import re
from collections import Counter, defaultdict
from typing import List, Dict, Tuple


class SimpleEmbedder:
    """
    Simple text embedding class using TF-IDF
    
    This class converts text documents into numerical vectors that can be
    compared for similarity. TF-IDF measures how important a word is to
    a document relative to a collection of documents.
    """
    
    def __init__(self):
        """Initialize the embedder"""
        self.vocabulary = set()  # All unique words across documents
        self.idf_scores = {}     # Inverse document frequency scores
        self.document_vectors = {}  # Stored document vectors
        self.is_fitted = False   # Whether the embedder has been trained
    
    def _preprocess_text(self, text: str) -> List[str]:
        """
        Clean and tokenize text into words
        
        Args:
            text (str): Raw text to preprocess
            
        Returns:
            List[str]: List of cleaned, lowercase words
        """
        # Convert to lowercase and remove special characters
        text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
        
        # Split into words and remove empty strings
        words = [word.strip() for word in text.split() if word.strip()]
        
        # Remove common stop words (simple list)
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 
                     'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were',
                     'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 
                     'did', 'will', 'would', 'could', 'should', 'may', 'might',
                     'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she',
                     'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'}
        
        words = [word for word in words if word not in stop_words and len(word) > 2]
        
        return words
    
    def fit(self, documents: List[Dict]) -> None:
        """
        Train the embedder on a collection of documents
        
        This method calculates IDF scores for all words in the corpus
        and prepares the embedder for creating document vectors.
        
        Args:
            documents (List[Dict]): List of document dictionaries with 'content' key
        """
        print("Training embedder on document corpus...")
        
        # Preprocess all documents and build vocabulary
        processed_docs = []
        word_doc_count = defaultdict(int)  # Count how many docs each word appears in
        
        for doc in documents:
            # Combine title and content for richer representation
            full_text = doc.get('title', '') + ' ' + doc.get('content', '')
            words = self._preprocess_text(full_text)
            processed_docs.append(words)
            
            # Count document frequency for each word
            unique_words_in_doc = set(words)
            for word in unique_words_in_doc:
                word_doc_count[word] += 1
                self.vocabulary.add(word)
        
        # Calculate IDF (Inverse Document Frequency) scores
        total_docs = len(documents)
        for word in self.vocabulary:
            # IDF = log(total_documents / documents_containing_word)
            self.idf_scores[word] = math.log(total_docs / word_doc_count[word])
        
        # Pre-compute document vectors for faster retrieval
        for i, (doc, words) in enumerate(zip(documents, processed_docs)):
            doc_id = doc.get('id', i)
            self.document_vectors[doc_id] = self._calculate_tfidf_vector(words)
        
        self.is_fitted = True
        print(f"Embedder trained! Vocabulary size: {len(self.vocabulary)}")
    
    def _calculate_tfidf_vector(self, words: List[str]) -> Dict[str, float]:
        """
        Calculate TF-IDF vector for a document
        
        Args:
            words (List[str]): Preprocessed words from document
            
        Returns:
            Dict[str, float]: TF-IDF vector as word -> score mapping
        """
        # Calculate term frequency (TF)
        word_count = Counter(words)
        total_words = len(words)
        
        tfidf_vector = {}
        for word in word_count:
            if word in self.vocabulary:
                # TF = (word_count / total_words)
                tf = word_count[word] / total_words
                # TF-IDF = TF * IDF
                tfidf_vector[word] = tf * self.idf_scores[word]
        
        return tfidf_vector
    
    def embed_text(self, text: str) -> Dict[str, float]:
        """
        Convert text to TF-IDF vector
        
        Args:
            text (str): Text to embed
            
        Returns:
            Dict[str, float]: TF-IDF vector
        """
        if not self.is_fitted:
            raise ValueError("Embedder must be fitted before embedding text")
        
        words = self._preprocess_text(text)
        return self._calculate_tfidf_vector(words)
    
    def calculate_similarity(self, vector1: Dict[str, float], vector2: Dict[str, float]) -> float:
        """
        Calculate cosine similarity between two TF-IDF vectors
        
        Cosine similarity measures the angle between two vectors,
        giving a score between 0 (completely different) and 1 (identical).
        
        Args:
            vector1 (Dict[str, float]): First TF-IDF vector
            vector2 (Dict[str, float]): Second TF-IDF vector
            
        Returns:
            float: Cosine similarity score (0-1)
        """
        # Get all unique words from both vectors
        all_words = set(vector1.keys()) | set(vector2.keys())
        
        # Calculate dot product and magnitudes
        dot_product = 0
        magnitude1 = 0
        magnitude2 = 0
        
        for word in all_words:
            score1 = vector1.get(word, 0)
            score2 = vector2.get(word, 0)
            
            dot_product += score1 * score2
            magnitude1 += score1 ** 2
            magnitude2 += score2 ** 2
        
        # Avoid division by zero
        if magnitude1 == 0 or magnitude2 == 0:
            return 0
        
        # Cosine similarity = dot_product / (magnitude1 * magnitude2)
        similarity = dot_product / (math.sqrt(magnitude1) * math.sqrt(magnitude2))
        return similarity
    
    def get_document_vector(self, doc_id: int) -> Dict[str, float]:
        """
        Get the pre-computed vector for a document
        
        Args:
            doc_id (int): Document ID
            
        Returns:
            Dict[str, float]: TF-IDF vector for the document
        """
        return self.document_vectors.get(doc_id, {})