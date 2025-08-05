"""
Fake Database Module for Simple RAG System

This module contains sample documents that simulate a knowledge base.
In a real-world scenario, this would be replaced with actual database connections
or document stores like Elasticsearch, Pinecone, or a vector database.
"""

# Sample documents representing different topics
# Each document has an ID, title, and content
FAKE_DOCUMENTS = [
    {
        "id": 1,
        "title": "Introduction to Machine Learning",
        "content": """
        Machine learning is a subset of artificial intelligence that enables computers 
        to learn and make decisions without being explicitly programmed. It involves 
        algorithms that can identify patterns in data and make predictions or decisions 
        based on those patterns. Common types include supervised learning, unsupervised 
        learning, and reinforcement learning. Applications range from image recognition 
        to natural language processing and recommendation systems.
        """
    },
    {
        "id": 2,
        "title": "Python Programming Basics",
        "content": """
        Python is a high-level, interpreted programming language known for its 
        simplicity and readability. It supports multiple programming paradigms 
        including procedural, object-oriented, and functional programming. Python 
        is widely used in web development, data science, artificial intelligence, 
        and automation. Key features include dynamic typing, automatic memory 
        management, and a vast standard library.
        """
    },
    {
        "id": 3,
        "title": "Data Science Fundamentals",
        "content": """
        Data science is an interdisciplinary field that combines statistics, 
        computer science, and domain expertise to extract insights from data. 
        The data science process typically involves data collection, cleaning, 
        exploration, modeling, and visualization. Common tools include Python, 
        R, SQL, and various machine learning libraries. Data scientists work 
        with structured and unstructured data to solve business problems.
        """
    },
    {
        "id": 4,
        "title": "Web Development with Flask",
        "content": """
        Flask is a lightweight web framework for Python that provides tools 
        and libraries for building web applications. It follows the WSGI 
        specification and is designed to be simple and flexible. Flask includes 
        features for routing, templating, and request handling. It's often used 
        for creating APIs, small to medium web applications, and prototyping. 
        Flask can be extended with various plugins for database integration, 
        authentication, and more.
        """
    },
    {
        "id": 5,
        "title": "Database Design Principles",
        "content": """
        Database design involves organizing data efficiently and ensuring data 
        integrity. Key principles include normalization to reduce redundancy, 
        defining proper relationships between tables, and choosing appropriate 
        data types. Good database design considers performance, scalability, 
        and maintainability. Common database types include relational (SQL) 
        and non-relational (NoSQL) databases, each suited for different use cases.
        """
    }
]

class FakeDatabase:
    """
    Simple in-memory database simulation
    
    This class provides basic CRUD operations for our fake documents.
    In a real RAG system, this would be replaced with actual database
    connections or vector database operations.
    """
    
    def __init__(self):
        """Initialize the database with sample documents"""
        self.documents = {doc["id"]: doc for doc in FAKE_DOCUMENTS}
    
    def get_all_documents(self):
        """
        Retrieve all documents from the database
        
        Returns:
            list: List of all document dictionaries
        """
        return list(self.documents.values())
    
    def get_document_by_id(self, doc_id):
        """
        Retrieve a specific document by its ID
        
        Args:
            doc_id (int): The document ID to retrieve
            
        Returns:
            dict or None: Document dictionary if found, None otherwise
        """
        return self.documents.get(doc_id)
    
    def search_documents_by_keyword(self, keyword):
        """
        Simple keyword search across all documents
        
        This is a basic search that looks for the keyword in title and content.
        In a real RAG system, this would be replaced with semantic search
        using embeddings and vector similarity.
        
        Args:
            keyword (str): Keyword to search for
            
        Returns:
            list: List of documents containing the keyword
        """
        keyword_lower = keyword.lower()
        matching_docs = []
        
        for doc in self.documents.values():
            # Search in both title and content (case-insensitive)
            if (keyword_lower in doc["title"].lower() or 
                keyword_lower in doc["content"].lower()):
                matching_docs.append(doc)
        
        return matching_docs
    
    def get_document_count(self):
        """
        Get the total number of documents in the database
        
        Returns:
            int: Number of documents
        """
        return len(self.documents)