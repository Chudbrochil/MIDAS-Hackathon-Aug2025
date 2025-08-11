"""
Incident Management System - Data Manager

Handles incident data ingestion, storage, and retrieval using vector embeddings
for semantic search capabilities. Supports multiple data formats and provides
comprehensive filtering and statistical analysis.

"""

from typing import List, Dict, Any, Union
import logging
import json
from pathlib import Path
import pandas as pd
from langchain.embeddings import OllamaEmbeddings
from langchain.schema import Document
from langchain_community.vectorstores import Chroma

from .data_models import Incident, IncidentPriority, Coordinates, IncidentStatistics


class DataManager:
    """
    Central data management class that handles incident data storage and retrieval.

    Features:
    - Vector-based similarity search using embeddings
    - Support for GeoJSON and other structured formats
    - Built-in data validation and cleaning
    - Statistical analysis and reporting
    - Flexible filtering capabilities
    """

    def __init__(self,
                 db_path: str = "./chroma_db",
                 embedding_model: str = "nomic-embed-text",
                 ollama_base_url: str = "http://localhost:11434"):
        """
        Initialize the data manager with vector store and embedding configuration.

        Args:
            db_path: Path for persistent vector database storage
            embedding_model: Name of the embedding model to use
            ollama_base_url: Base URL for Ollama API service
        """
        self.incidents: List[Incident] = []
        self.db_path = Path(db_path)
        self.db_path.mkdir(exist_ok=True)

        # Initialize embedding service
        self.embeddings = OllamaEmbeddings(
            model=embedding_model,
            base_url=ollama_base_url
        )

        # Setup persistent vector store for semantic search
        self.vectorstore = Chroma(
            collection_name="incidents",
            embedding_function=self.embeddings,
            persist_directory=str(self.db_path)
        )

        logging.info(f"Initialized DataManager with database at {self.db_path}")

    def load_from_geojson_file(self, filepath: str) -> int:
        """
        Load incident data from a GeoJSON file.

        Args:
            filepath: Path to the GeoJSON file

        Returns:
            Number of incidents successfully loaded
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                data = json.load(file)
            return self.load_from_geojson(data)
        except FileNotFoundError:
            logging.error(f"File not found: {filepath}")
            raise
        except json.JSONDecodeError as e:
            logging.error(f"Invalid JSON in file {filepath}: {e}")
            raise

    def load_from_geojson(self, geojson_data: Dict[str, Any]) -> int:
        """
        Process GeoJSON data and convert to incident objects.

        This method handles various GeoJSON schemas and provides flexible
        field mapping to accommodate different data sources.

        Args:
            geojson_data: GeoJSON FeatureCollection dictionary

        Returns:
            Number of incidents successfully processed
        """
        loaded_count = 0
        documents = []
        failed_count = 0

        features = geojson_data.get('features', [])
        logging.info(f"Processing {len(features)} features from GeoJSON data")

        for feature in features:
            try:
                props = feature.get('properties', {})
                geometry = feature.get('geometry', {})
                coords = geometry.get('coordinates', [0, 0])

                # Create incident with flexible field mapping
                incident = Incident(
                    incident_entry_id=props.get('incident_entry_id'),
                    incident_id=props.get('incident_id'),
                    incident_location=props.get('incident_location', ''),
                    coordinates=Coordinates(
                        longitude=coords[0] if len(coords) > 0 else 0,
                        latitude=coords[1] if len(coords) > 1 else 0
                    ),
                    postal_code=props.get('postal_code', props.get('zip_code', '')),
                    service_area=props.get('service_area', props.get('precinct', 1)),
                    district=props.get('district', props.get('scout_car_area', '')),
                    neighborhood_name=props.get('neighborhood_name', ''),
                    administrative_district=props.get('administrative_district',
                                                      props.get('council_district', 1)),
                    incident_source=props.get('incident_source', props.get('call_source', '')),
                    incident_description=props.get('incident_description',
                                                   props.get('call_description', '')),
                    category=props.get('category', ''),
                    priority=self._normalize_priority(props.get('priority', 'Medium')),
                    incident_type=props.get('incident_type', props.get('call_group', '')),
                    incident_code=props.get('incident_code', props.get('call_code', '')),
                    reported_at=props.get('reported_at', props.get('called_at')),
                    intake_time=props.get('intake_time', 0),
                    dispatch_time=props.get('dispatch_time', 0),
                    travel_time=props.get('travel_time', 0),
                    on_scene_time=props.get('on_scene_time', 0),
                    total_response_time=props.get('total_response_time', 0),
                    total_time=props.get('total_time', 0),
                    external_id=props.get('external_id', props.get('ESRI_OID'))
                )

                self.incidents.append(incident)
                documents.append(incident.to_langchain_document())
                loaded_count += 1

            except Exception as e:
                failed_count += 1
                logging.warning(f"Failed to process incident feature: {e}")
                continue

        # Bulk add to vector store for efficiency
        if documents:
            self.vectorstore.add_documents(documents)
            self.vectorstore.persist()
            logging.info(f"Successfully indexed {len(documents)} incidents in vector database")

        if failed_count > 0:
            logging.warning(f"Failed to process {failed_count} incidents due to data issues")

        return loaded_count

    def _normalize_priority(self, priority_value: Union[str, int]) -> IncidentPriority:
        """
        Convert various priority representations to standardized enum.

        Handles numeric codes, text descriptions, and mixed formats
        commonly found in different incident management systems.

        Args:
            priority_value: Priority value in various formats

        Returns:
            Standardized IncidentPriority enum value
        """
        if isinstance(priority_value, (int, float)) or str(priority_value).isdigit():
            priority_map = {
                1: IncidentPriority.CRITICAL,
                2: IncidentPriority.HIGH,
                3: IncidentPriority.MEDIUM,
                4: IncidentPriority.LOW
            }
            return priority_map.get(int(priority_value), IncidentPriority.MEDIUM)

        priority_str = str(priority_value).lower().strip()

        # Handle common text variations
        if priority_str in ['critical', 'emergency', 'urgent', '1']:
            return IncidentPriority.CRITICAL
        elif priority_str in ['high', 'important', '2']:
            return IncidentPriority.HIGH
        elif priority_str in ['medium', 'normal', 'routine', '3']:
            return IncidentPriority.MEDIUM
        else:
            return IncidentPriority.LOW

    def load_from_file(self, filepath: str) -> int:
        """
        Universal file loader that detects format and delegates to appropriate handler.

        Args:
            filepath: Path to the data file

        Returns:
            Number of incidents loaded

        Raises:
            ValueError: If file format is not supported
        """
        filepath = Path(filepath)

        if filepath.suffix.lower() in ['.json', '.geojson']:
            return self.load_from_geojson_file(str(filepath))
        else:
            supported_formats = ['.json', '.geojson']
            raise ValueError(f"Unsupported file format: {filepath.suffix}. "
                             f"Supported formats: {supported_formats}")

    def similarity_search(self, query: str, k: int = 10) -> List[Document]:
        """
        Perform semantic similarity search across incident descriptions.

        Args:
            query: Natural language search query
            k: Maximum number of results to return

        Returns:
            List of relevant documents ordered by similarity
        """
        return self.vectorstore.similarity_search(query=query, k=k)

    def similarity_search_with_scores(self, query: str, k: int = 10) -> List[tuple]:
        """
        Similarity search with relevance scores for result ranking.

        Args:
            query: Natural language search query
            k: Maximum number of results to return

        Returns:
            List of tuples containing (document, similarity_score)
        """
        return self.vectorstore.similarity_search_with_score(query, k=k)

    def get_incidents_by_ids(self, incident_ids: List[str]) -> List[Incident]:
        """
        Retrieve specific incidents by their ID values.

        Args:
            incident_ids: List of incident ID strings

        Returns:
            List of matching Incident objects
        """
        return [incident for incident in self.incidents
                if incident.incident_id in incident_ids]

    def filter_incidents(self, **filters) -> List[Incident]:
        """
        Apply multiple filters to incident data with flexible criteria.

        Supported filters:
        - priority: Single priority or list of priorities
        - service_area: Single area or list of areas
        - category: Single category or list of categories
        - neighborhood: Single neighborhood or list of neighborhoods

        Args:
            **filters: Filter criteria as keyword arguments

        Returns:
            Filtered list of incidents
        """
        filtered = self.incidents.copy()

        # Priority filter
        if 'priority' in filters:
            priorities = (filters['priority'] if isinstance(filters['priority'], list)
                          else [filters['priority']])
            filtered = [inc for inc in filtered if inc.priority in priorities]

        # Service area filter
        if 'service_area' in filters:
            service_areas = (filters['service_area'] if isinstance(filters['service_area'], list)
                             else [filters['service_area']])
            filtered = [inc for inc in filtered if inc.service_area in service_areas]

        # Category filter
        if 'category' in filters:
            categories = (filters['category'] if isinstance(filters['category'], list)
                          else [filters['category']])
            filtered = [inc for inc in filtered if inc.category in categories]

        # Neighborhood filter
        if 'neighborhood' in filters:
            neighborhoods = (filters['neighborhood'] if isinstance(filters['neighborhood'], list)
                             else [filters['neighborhood']])
            filtered = [inc for inc in filtered if inc.neighborhood_name in neighborhoods]

        return filtered

    def calculate_statistics(self) -> IncidentStatistics:
        """
        Generate comprehensive statistical analysis of current incident data.

        Returns:
            IncidentStatistics object with aggregated metrics and distributions
        """
        if not self.incidents:
            return IncidentStatistics(
                total_incidents=0,
                avg_response_time=0,
                avg_total_time=0,
                priority_distribution={},
                category_distribution={},
                service_area_distribution={},
                response_time_categories={}
            )

        total_count = len(self.incidents)

        # Calculate timing averages
        avg_response = sum(inc.total_response_time for inc in self.incidents) / total_count
        avg_total = sum(inc.total_time for inc in self.incidents) / total_count

        # Build distribution dictionaries
        priority_dist = {}
        category_dist = {}
        service_area_dist = {}
        response_categories = {}

        for incident in self.incidents:
            # Priority distribution
            priority_key = incident.priority.value
            priority_dist[priority_key] = priority_dist.get(priority_key, 0) + 1

            # Category distribution
            category_dist[incident.category] = category_dist.get(incident.category, 0) + 1

            # Service area distribution
            service_area_dist[incident.service_area] = service_area_dist.get(incident.service_area, 0) + 1

            # Response time performance categories
            response_cat = incident.response_category
            response_categories[response_cat] = response_categories.get(response_cat, 0) + 1

        return IncidentStatistics(
            total_incidents=total_count,
            avg_response_time=avg_response,
            avg_total_time=avg_total,
            priority_distribution=priority_dist,
            category_distribution=category_dist,
            service_area_distribution=service_area_dist,
            response_time_categories=response_categories
        )

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert incident data to pandas DataFrame for analysis and export.

        Returns:
            DataFrame with flattened incident data and computed columns
        """
        if not self.incidents:
            return pd.DataFrame()

        records = []
        for incident in self.incidents:
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
                'is_high_priority': incident.is_high_priority
            })

        return pd.DataFrame(records)

    def get_summary(self) -> Dict[str, Any]:
        """
        Generate a high-level summary of the current dataset.

        Returns:
            Dictionary containing key metrics and data quality indicators
        """
        if not self.incidents:
            return {"status": "No data loaded"}

        stats = self.calculate_statistics()
        df = self.to_dataframe()

        return {
            "total_incidents": len(self.incidents),
            "date_range": {
                "earliest": df['reported_at'].min().isoformat() if not df.empty else None,
                "latest": df['reported_at'].max().isoformat() if not df.empty else None
            },
            "performance_metrics": {
                "avg_response_time_minutes": round(stats.avg_response_time, 2),
                "excellent_response_rate": round(
                    stats.response_time_categories.get('Excellent', 0) / len(self.incidents) * 100, 1
                )
            },
            "data_coverage": {
                "unique_neighborhoods": len(set(inc.neighborhood_name for inc in self.incidents)),
                "unique_categories": len(set(inc.category for inc in self.incidents)),
                "service_areas": len(set(inc.service_area for inc in self.incidents))
            }
        }