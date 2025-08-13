"""
Query Processor

Provides natural language query processing using Retrieval-Augmented Generation (RAG)
to answer questions about incident data. Integrates with vector search and language
models to deliver contextual, data-driven responses.

"""

import logging
from typing import Dict, Any, List, Optional
import pandas as pd
import requests
from .data_manager import DataManager
from .data_models import Incident


class QueryProcessor:
    """
    Processes natural language queries about incident data using RAG methodology.

    Combines semantic search capabilities with language model generation to provide
    accurate, contextual responses backed by actual incident data.
    """

    def __init__(self,
                 data_manager: DataManager,
                 ollama_base_url: str = "http://localhost:11434",
                 model_name: str = "llama3.2"):
        """
        Initialize the query processor with data source and model configuration.

        Args:
            data_manager: DataManager instance for incident data access
            ollama_base_url: Base URL for Ollama API service
            model_name: Name of the language model to use for generation
        """
        self.data_manager = data_manager
        self.ollama_base_url = ollama_base_url.rstrip('/')
        self.model_name = model_name

        logging.info(f"Initialized QueryProcessor with model: {model_name}")

    def process_query(self, query: str, max_incidents: int = 10) -> Dict[str, Any]:
        """
        Process a natural language query and return comprehensive results.

        This method implements the RAG pattern:
        1. Retrieve relevant incidents using semantic search
        2. Augment query with contextual data
        3. Generate response using language model
        4. Prepare data for visualization

        Args:
            query: Natural language question about incidents
            max_incidents: Maximum number of incidents to consider

        Returns:
            Dictionary containing response text, data, and metadata
        """
        try:
            logging.info(f"Processing query: {query[:100]}...")

            # Retrieve relevant documents using vector search
            search_results = self.data_manager.similarity_search(query, k=max_incidents)

            # Extract incident IDs from search results metadata
            relevant_incident_ids = []
            for doc in search_results:
                if 'incident_id' in doc.metadata:
                    relevant_incident_ids.append(doc.metadata['incident_id'])

            # Get full incident objects
            relevant_incidents = self.data_manager.get_incidents_by_ids(relevant_incident_ids)

            # Build context for language model
            context = self._build_context(relevant_incidents, query)

            # Generate natural language response
            response_text = self._generate_response(query, context)

            # Prepare visualization data
            visualization_data = self._prepare_visualization_data(relevant_incidents, query)

            # Compile analytical insights
            insights = self._extract_insights(relevant_incidents, query)

            return {
                "text": response_text,
                "data": visualization_data,
                "relevant_incidents": relevant_incidents,
                "search_results": search_results,
                "insights": insights,
                "query_metadata": {
                    "incidents_found": len(relevant_incidents),
                    "search_terms": query,
                    "model_used": self.model_name
                }
            }

        except Exception as e:
            logging.error(f"Error processing query '{query}': {e}")
            return {
                "text": f"I encountered an error while processing your query: {str(e)}",
                "data": None,
                "relevant_incidents": [],
                "search_results": [],
                "insights": {},
                "query_metadata": {"error": str(e)}
            }

    def _build_context(self, incidents: List[Incident], query: str) -> str:
        """
        Create comprehensive context from relevant incidents and database statistics.

        Args:
            incidents: List of relevant incidents
            query: Original user query for context

        Returns:
            Formatted context string for language model
        """
        if not incidents:
            return "No directly relevant incidents found in the database."

        context_sections = []

        # Add database overview for perspective
        stats = self.data_manager.calculate_statistics()
        context_sections.append(f"""
        DATABASE OVERVIEW:
        Total incidents in database: {stats.total_incidents:,}
        Average response time: {stats.avg_response_time:.1f} minutes
        Response time distribution: {dict(list(stats.response_time_categories.items()))}
        Priority breakdown: {dict(list(stats.priority_distribution.items()))}
        """)

        # Add details for most relevant incidents
        context_sections.append("RELEVANT INCIDENTS FOR YOUR QUERY:")

        for idx, incident in enumerate(incidents[:8], 1):  # Limit to top 8 for context size
            context_sections.append(f"""
            {idx}. Incident {incident.incident_id}
               • Location: {incident.incident_location} ({incident.neighborhood_name})
               • Description: {incident.incident_description[:200]}...
               • Category: {incident.category} | Priority: {incident.priority.value}
               • Response Time: {incident.total_response_time:.1f} minutes ({incident.response_category})
               • Date: {incident.reported_at.strftime('%Y-%m-%d %H:%M')}
               • Service Area: {incident.service_area}
            """)

        # Add summary statistics for the filtered set
        if len(incidents) > 1:
            avg_response = sum(inc.total_response_time for inc in incidents) / len(incidents)
            high_priority_count = sum(1 for inc in incidents if inc.is_high_priority)

            context_sections.append(f"""
            SUMMARY OF RELEVANT INCIDENTS:
            • Found {len(incidents)} incidents matching your query
            • Average response time: {avg_response:.1f} minutes
            • High priority incidents: {high_priority_count}
            • Neighborhoods covered: {len(set(inc.neighborhood_name for inc in incidents))}
            """)

        return "\n".join(context_sections)

    def _generate_response(self, query: str, context: str) -> str:
        """
        Generate natural language response using the configured language model.

        Args:
            query: User's original question
            context: Contextual information from incident data

        Returns:
            Generated response text
        """
        system_prompt = """You are an expert data analyst specializing in emergency response and incident management. 

        Your role is to help users understand patterns, trends, and insights from incident data. You should:

        • Provide clear, actionable insights based on the data
        • Highlight performance metrics and operational patterns  
        • Identify potential areas for improvement
        • Use specific numbers and examples from the data
        • Explain findings in terms relevant to emergency management
        • Be concise but comprehensive in your analysis

        If the data doesn't support a conclusion, state that clearly rather than speculating."""

        prompt = f"""
        SYSTEM CONTEXT: {system_prompt}

        INCIDENT DATA CONTEXT:
        {context}

        USER QUESTION: {query}

        Please analyze the incident data and provide a detailed response that directly addresses the user's question. 
        Include specific metrics, patterns, and actionable insights based on the available data.
        """

        try:
            response = requests.post(
                f"{self.ollama_base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,  # Low temperature for factual responses
                        "top_p": 0.9,
                        "max_tokens": 800,
                        "stop": ["USER:", "SYSTEM:"]
                    }
                },
                timeout=45,
                headers={"Content-Type": "application/json"}
            )

            if response.status_code == 200:
                result = response.json()
                return result.get('response', 'No response generated from the model.')
            else:
                logging.error(f"Ollama API error: HTTP {response.status_code}")
                return f"Error communicating with language model (HTTP {response.status_code}). Please check your Ollama service."

        except requests.exceptions.Timeout:
            return "Request timed out. The language model may be busy or unavailable."
        except requests.exceptions.ConnectionError:
            return "Unable to connect to language model service. Please verify Ollama is running and accessible."
        except Exception as e:
            logging.error(f"Unexpected error in response generation: {e}")
            return f"Unexpected error occurred while generating response: {str(e)}"

    def _prepare_visualization_data(self, incidents: List[Incident], query: str) -> Optional[pd.DataFrame]:
        """
        Prepare incident data for visualization based on query context.

        Args:
            incidents: List of relevant incidents
            query: Original query to determine optimal data structure

        Returns:
            DataFrame optimized for the specific query type, or None if no data
        """
        if not incidents:
            return None

        # Convert incidents to structured data
        records = []
        for incident in incidents:
            records.append({
                'incident_id': incident.incident_id,
                'incident_location': incident.incident_location,
                'neighborhood_name': incident.neighborhood_name,
                'incident_description': incident.incident_description,
                'category': incident.category,
                'priority': incident.priority.value,
                'service_area': incident.service_area,
                'reported_at': incident.reported_at,
                'total_response_time': incident.total_response_time,
                'total_time': incident.total_time,
                'latitude': incident.coordinates.latitude,
                'longitude': incident.coordinates.longitude,
                'postal_code': incident.postal_code,
                'administrative_district': incident.administrative_district,
                'response_category': incident.response_category,
                'is_high_priority': incident.is_high_priority,
                'hour_of_day': incident.reported_at.hour,
                'day_of_week': incident.reported_at.strftime('%A'),
                'month': incident.reported_at.strftime('%B')
            })

        df = pd.DataFrame(records)

        # Add computed columns for enhanced analysis
        df['response_time_bucket'] = pd.cut(
            df['total_response_time'],
            bins=[0, 8, 15, 25, float('inf')],
            labels=['Excellent', 'Good', 'Fair', 'Poor']
        )

        return df

    def _extract_insights(self, incidents: List[Incident], query: str) -> Dict[str, Any]:
        """
        Generate analytical insights specific to the query and incident data.

        Args:
            incidents: Relevant incidents for analysis
            query: Original query for context

        Returns:
            Dictionary containing key insights and metrics
        """
        if not incidents:
            return {"message": "No incidents available for analysis"}

        insights = {}
        df = pd.DataFrame([{
            'category': inc.category,
            'priority': inc.priority.value,
            'neighborhood': inc.neighborhood_name,
            'service_area': inc.service_area,
            'response_time': inc.total_response_time,
            'response_category': inc.response_category,
            'hour': inc.reported_at.hour,
            'day_of_week': inc.reported_at.strftime('%A')
        } for inc in incidents])

        # Response time analysis
        insights['response_metrics'] = {
            'average_response_time': round(df['response_time'].mean(), 2),
            'median_response_time': round(df['response_time'].median(), 2),
            'fastest_response': round(df['response_time'].min(), 2),
            'slowest_response': round(df['response_time'].max(), 2),
            'performance_distribution': df['response_category'].value_counts().to_dict()
        }

        # Geographic distribution
        insights['geographic_patterns'] = {
            'neighborhoods_affected': df['neighborhood'].nunique(),
            'service_areas_involved': df['service_area'].nunique(),
            'top_neighborhoods': df['neighborhood'].value_counts().head(5).to_dict(),
            'busiest_service_areas': df['service_area'].value_counts().head(5).to_dict()
        }

        # Category and priority analysis
        insights['incident_patterns'] = {
            'category_breakdown': df['category'].value_counts().to_dict(),
            'priority_distribution': df['priority'].value_counts().to_dict(),
            'high_priority_percentage': round(
                len(df[df['priority'].isin(['Critical', 'High'])]) / len(df) * 100, 1
            )
        }

        # Temporal patterns
        insights['temporal_patterns'] = {
            'peak_hours': df['hour'].value_counts().head(3).to_dict(),
            'busiest_days': df['day_of_week'].value_counts().to_dict()
        }

        return insights

    def get_query_suggestions(self, category: str = "general") -> List[str]:
        """
        Provide sample queries based on available data and common analysis needs.

        Args:
            category: Type of suggestions to generate

        Returns:
            List of suggested query strings
        """
        suggestions = {
            "performance": [
                "What is the average response time by neighborhood?",
                "Which service areas have the fastest response times?",
                "How many incidents meet the excellent response time standard?",
                "What factors correlate with longer response times?"
            ],
            "patterns": [
                "What are the most common incident types?",
                "Which neighborhoods have the highest incident volume?",
                "How do incident patterns vary by time of day?",
                "What is the seasonal trend in incident categories?"
            ],
            "operations": [
                "Which service areas are experiencing high workload?",
                "How can we optimize resource allocation?",
                "What are the main drivers of response time delays?",
                "Which incident types require the most on-scene time?"
            ],
            "general": [
                "Show me incident trends over the past month",
                "What are the key performance indicators?",
                "Which areas need additional resources?",
                "How do we compare to response time benchmarks?"
            ]
        }

        return suggestions.get(category, suggestions["general"])

    def validate_query(self, query: str) -> Dict[str, Any]:
        """
        Validate and analyze query before processing.

        Args:
            query: User query string

        Returns:
            Validation results and suggestions
        """
        validation = {
            "is_valid": True,
            "confidence": "high",
            "suggestions": [],
            "estimated_results": 0
        }

        # Basic validation checks
        if not query or len(query.strip()) < 5:
            validation["is_valid"] = False
            validation["suggestions"].append("Please provide a more detailed question")
            return validation

        # Estimate potential results
        quick_search = self.data_manager.similarity_search(query, k=5)
        validation["estimated_results"] = len(quick_search)

        if validation["estimated_results"] == 0:
            validation["confidence"] = "low"
            validation["suggestions"].append("Try using different keywords or broader terms")

        return validation