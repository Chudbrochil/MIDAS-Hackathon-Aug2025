"""
Text Generation Module for RAG System

This module handles the "Generation" part of RAG (Retrieval-Augmented Generation).
It takes a user query and relevant context documents to generate a comprehensive answer.

Since we're building a simple system without external LLM APIs, this module uses
template-based generation with intelligent text extraction and summarization.

In a production RAG system, this would typically use:
- OpenAI GPT models
- Anthropic Claude
- Local models like Llama, Mistral
- Or other transformer-based language models
"""

import re
from typing import List, Dict, Optional


class SimpleGenerator:
    """
    Simple rule-based text generator for RAG responses
    
    This generator creates answers by intelligently combining information
    from retrieved documents using templates and pattern matching.
    """
    
    def __init__(self):
        """Initialize the generator with response templates"""
        
        # Response templates for different types of queries
        self.templates = {
            'definition': [
                "Based on the retrieved information, {concept} is {definition}.",
                "According to the knowledge base, {concept} can be defined as {definition}.",
                "{concept} refers to {definition}."
            ],
            'how_to': [
                "To {action}, you should {steps}.",
                "Here's how to {action}: {steps}",
                "The process for {action} involves {steps}."
            ],
            'comparison': [
                "When comparing {concept1} and {concept2}, {comparison}.",
                "{concept1} differs from {concept2} in that {comparison}.",
                "The main differences between {concept1} and {concept2} are {comparison}."
            ],
            'general': [
                "Based on the available information, {answer}.",
                "According to the retrieved documents, {answer}.",
                "From the knowledge base, {answer}."
            ]
        }
    
    def generate_response(self, query: str, context: str, retrieved_docs: List[Dict]) -> str:
        """
        Generate a response based on query and retrieved context
        
        Args:
            query (str): Original user query
            context (str): Context from retrieved documents
            retrieved_docs (List[Dict]): List of retrieved documents with scores
            
        Returns:
            str: Generated response
        """
        print(f"Generating response for query: '{query}'")
        
        # If no context available, return a default response
        if not context.strip() or not retrieved_docs:
            return self._generate_no_context_response(query)
        
        # Analyze query type to choose appropriate generation strategy
        query_type = self._classify_query(query)
        print(f"Detected query type: {query_type}")
        
        # Extract key information from context
        key_info = self._extract_key_information(context, query)
        
        # Generate response based on query type
        if query_type == 'definition':
            response = self._generate_definition_response(query, key_info, retrieved_docs)
        elif query_type == 'how_to':
            response = self._generate_how_to_response(query, key_info, retrieved_docs)
        elif query_type == 'comparison':
            response = self._generate_comparison_response(query, key_info, retrieved_docs)
        else:
            response = self._generate_general_response(query, key_info, retrieved_docs)
        
        # Add source attribution
        response = self._add_source_attribution(response, retrieved_docs)
        
        return response
    
    def _classify_query(self, query: str) -> str:
        """
        Classify the type of query to choose appropriate response strategy
        
        Args:
            query (str): User query
            
        Returns:
            str: Query type ('definition', 'how_to', 'comparison', 'general')
        """
        query_lower = query.lower()
        
        # Definition queries
        if any(word in query_lower for word in ['what is', 'define', 'definition', 'meaning']):
            return 'definition'
        
        # How-to queries
        if any(word in query_lower for word in ['how to', 'how do', 'how can', 'steps', 'process']):
            return 'how_to'
        
        # Comparison queries
        if any(word in query_lower for word in ['difference', 'compare', 'versus', 'vs', 'better']):
            return 'comparison'
        
        return 'general'
    
    def _extract_key_information(self, context: str, query: str) -> Dict[str, str]:
        """
        Extract key information from context relevant to the query
        
        Args:
            context (str): Retrieved document context
            query (str): Original query
            
        Returns:
            Dict[str, str]: Extracted key information
        """
        # Split context into sentences
        sentences = re.split(r'[.!?]+', context)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Extract query keywords
        query_words = set(re.findall(r'\b\w+\b', query.lower()))
        query_words = {word for word in query_words if len(word) > 3}
        
        # Find sentences most relevant to query
        relevant_sentences = []
        for sentence in sentences:
            sentence_words = set(re.findall(r'\b\w+\b', sentence.lower()))
            overlap = len(query_words & sentence_words)
            if overlap > 0:
                relevant_sentences.append((sentence, overlap))
        
        # Sort by relevance and take top sentences
        relevant_sentences.sort(key=lambda x: x[1], reverse=True)
        top_sentences = [s[0] for s in relevant_sentences[:3]]
        
        return {
            'key_sentences': top_sentences,
            'context_summary': ' '.join(top_sentences),
            'query_keywords': list(query_words)
        }
    
    def _generate_definition_response(self, query: str, key_info: Dict, retrieved_docs: List[Dict]) -> str:
        """Generate response for definition queries"""
        
        # Extract the concept being defined
        query_lower = query.lower()
        concept = query_lower.replace('what is', '').replace('define', '').replace('definition of', '').strip()
        
        # Use key sentences as definition
        definition = key_info['context_summary']
        
        # Choose a template and format it
        template = self.templates['definition'][0]
        response = template.format(concept=concept, definition=definition)
        
        return response
    
    def _generate_how_to_response(self, query: str, key_info: Dict, retrieved_docs: List[Dict]) -> str:
        """Generate response for how-to queries"""
        
        # Extract the action from query
        action = query.lower().replace('how to', '').replace('how do i', '').replace('how can i', '').strip()
        
        # Use key information as steps
        steps = key_info['context_summary']
        
        template = self.templates['how_to'][0]
        response = template.format(action=action, steps=steps)
        
        return response
    
    def _generate_comparison_response(self, query: str, key_info: Dict, retrieved_docs: List[Dict]) -> str:
        """Generate response for comparison queries"""
        
        # This is a simplified comparison - in practice, you'd need more sophisticated NLP
        comparison = key_info['context_summary']
        
        template = self.templates['comparison'][0]
        response = template.format(concept1="the topics", concept2="mentioned", comparison=comparison)
        
        return response
    
    def _generate_general_response(self, query: str, key_info: Dict, retrieved_docs: List[Dict]) -> str:
        """Generate response for general queries"""
        
        answer = key_info['context_summary']
        
        template = self.templates['general'][0]
        response = template.format(answer=answer)
        
        return response
    
    def _generate_no_context_response(self, query: str) -> str:
        """Generate response when no relevant context is found"""
        
        return (f"I don't have enough information in my knowledge base to answer your question "
                f"about '{query}'. Please try rephrasing your question or asking about a topic "
                f"covered in the available documents.")
    
    def _add_source_attribution(self, response: str, retrieved_docs: List[Dict]) -> str:
        """
        Add source attribution to the response
        
        Args:
            response (str): Generated response
            retrieved_docs (List[Dict]): Retrieved documents used
            
        Returns:
            str: Response with source attribution
        """
        if not retrieved_docs:
            return response
        
        # Add source information
        sources = []
        for item in retrieved_docs:
            doc = item['document']
            score = item['similarity_score']
            sources.append(f"'{doc['title']}' (relevance: {score:.2f})")
        
        attribution = f"\n\nSources: {', '.join(sources)}"
        
        return response + attribution
    
    def generate_with_explanation(self, query: str, context: str, retrieved_docs: List[Dict]) -> Dict[str, str]:
        """
        Generate response with detailed explanation of the generation process
        
        Args:
            query (str): User query
            context (str): Retrieved context
            retrieved_docs (List[Dict]): Retrieved documents
            
        Returns:
            Dict[str, str]: Response with generation explanation
        """
        print(f"\n{'='*60}")
        print("GENERATION PROCESS EXPLANATION")
        print(f"{'='*60}")
        
        print(f"\n1. QUERY ANALYSIS:")
        query_type = self._classify_query(query)
        print(f"   Query: '{query}'")
        print(f"   Detected type: {query_type}")
        
        print(f"\n2. CONTEXT PROCESSING:")
        key_info = self._extract_key_information(context, query)
        print(f"   Key sentences extracted: {len(key_info['key_sentences'])}")
        print(f"   Query keywords: {key_info['query_keywords']}")
        
        print(f"\n3. RESPONSE GENERATION:")
        response = self.generate_response(query, context, retrieved_docs)
        print(f"   Template used: {query_type}")
        print(f"   Final response length: {len(response)} characters")
        
        print(f"\n4. FINAL RESPONSE:")
        print(f"   {response}")
        
        print(f"{'='*60}\n")
        
        return {
            'response': response,
            'query_type': query_type,
            'key_info': str(key_info),
            'generation_strategy': f"Used {query_type} template with extracted key information"
        }