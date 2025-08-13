"""
Core Data Models

This module defines the foundational data structures for incident management,
including validation rules, enums, and conversion utilities for various
data interchange formats.

"""

from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any
from geopy.distance import geodesic
from langchain.schema import Document
from pydantic import BaseModel, Field, validator, root_validator


class IncidentPriority(str, Enum):
    """
    Standardized incident priority levels following emergency response protocols.
    """
    CRITICAL = "Critical"
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


class IncidentSource(str, Enum):
    """
    Categorizes the various channels through which incidents can be reported.
    """
    EMERGENCY_LINE = "Emergency Line"
    WALK_IN = "Walk-in"
    RADIO = "Radio"
    ONLINE = "Online"
    MOBILE_APP = "Mobile App"
    OTHER = "Other"


class Coordinates(BaseModel):
    """
    Geographic coordinate representation with validation and distance calculations.

    Attributes:
        longitude: Longitude coordinate (-180 to 180)
        latitude: Latitude coordinate (-90 to 90)
    """
    longitude: float = Field(..., ge=-180, le=180, description="Longitude in decimal degrees")
    latitude: float = Field(..., ge=-90, le=90, description="Latitude in decimal degrees")

    def distance_to(self, other: 'Coordinates') -> float:
        """
        Calculate the great-circle distance between two coordinate points.

        Args:
            other: Target coordinates for distance calculation

        Returns:
            Distance in kilometers using the geodesic method
        """
        return geodesic(
            (self.latitude, self.longitude),
            (other.latitude, other.longitude)
        ).kilometers


class Incident(BaseModel):
    """
    Comprehensive incident data model with built-in validation and utilities.

    This model handles incident data from various sources and provides
    standardized access to incident information, timing metrics, and
    geographic data.
    """

    @root_validator(pre=True)
    def sanitize_timing_data(cls, values):
        """
        Ensures all timing fields contain valid positive values.
        Negative times are converted to zero as they indicate data quality issues.
        """
        time_fields = [
            "intake_time", "dispatch_time", "travel_time",
            "on_scene_time", "total_response_time", "total_time"
        ]
        for field in time_fields:
            if field in values and values[field] < 0:
                values[field] = 0.0
        return values

    # Core identifiers
    incident_entry_id: int = Field(..., description="Unique database entry identifier")
    incident_id: str = Field(..., description="Human-readable incident reference number")

    # Geographic information
    incident_location: str = Field(..., description="Street address or location description")
    coordinates: Coordinates = Field(..., description="GPS coordinates of incident")
    postal_code: str = Field(..., description="ZIP/postal code of incident location")
    service_area: int = Field(..., gt=0, description="Service area or precinct number")
    district: str = Field(..., description="Administrative district designation")
    neighborhood_name: str = Field(..., description="Neighborhood or community name")
    administrative_district: int = Field(..., gt=0, description="Administrative district number")

    # Incident classification
    incident_source: str = Field(..., description="How the incident was reported")
    incident_description: str = Field(..., description="Detailed description of the incident")
    category: str = Field(..., description="Primary incident category")
    priority: IncidentPriority = Field(..., description="Assigned priority level")
    incident_type: str = Field(..., description="Specific incident type classification")
    incident_code: str = Field(..., description="Internal incident code")

    # Temporal data (all times in minutes)
    reported_at: datetime = Field(..., description="When the incident was first reported")
    intake_time: float = Field(..., ge=0, description="Time from report to intake completion")
    dispatch_time: float = Field(..., ge=0, description="Time from intake to dispatch")
    travel_time: float = Field(..., ge=0, description="Time from dispatch to arrival")
    on_scene_time: float = Field(..., ge=0, description="Time spent on scene")
    total_response_time: float = Field(..., ge=0, description="Total time from report to arrival")
    total_time: float = Field(..., ge=0, description="Total time from report to completion")

    # Optional fields
    external_id: Optional[int] = Field(None, description="External system reference ID")

    @validator('reported_at', pre=True)
    def parse_datetime_string(cls, value):
        """
        Handles datetime parsing from various string formats including ISO format.
        """
        if isinstance(value, str):
            # Handle ISO format with Z timezone indicator
            return datetime.fromisoformat(value.replace('Z', '+00:00'))
        return value

    @property
    def is_high_priority(self) -> bool:
        """
        Quick check for high-priority incidents requiring immediate attention.

        Returns:
            True if incident is Critical or High priority
        """
        return self.priority in [IncidentPriority.CRITICAL, IncidentPriority.HIGH]

    @property
    def response_category(self) -> str:
        """
        Categorizes response performance based on total response time.

        Performance thresholds:
        - Excellent: ≤ 8 minutes
        - Good: ≤ 15 minutes  
        - Fair: ≤ 25 minutes
        - Poor: > 25 minutes

        Returns:
            Performance category as string
        """
        if self.total_response_time <= 8:
            return "Excellent"
        elif self.total_response_time <= 15:
            return "Good"
        elif self.total_response_time <= 25:
            return "Fair"
        return "Poor"

    def to_document_text(self) -> str:
        """
        Converts incident data to structured text format for search indexing.

        Returns:
            Formatted text representation optimized for full-text search
        """
        return f"""
        Incident ID: {self.incident_id}
        Location: {self.incident_location}
        Neighborhood: {self.neighborhood_name}
        Service Area: {self.service_area}
        Description: {self.incident_description}
        Category: {self.category}
        Priority: {self.priority}
        Date: {self.reported_at.strftime('%Y-%m-%d %H:%M')}
        Response Time: {self.total_response_time} minutes
        Postal Code: {self.postal_code}
        Administrative District: {self.administrative_district}
        """

    def to_langchain_document(self) -> Document:
        """
        Converts incident to LangChain Document format for vector search.

        Returns:
            Document object with content and structured metadata
        """
        return Document(
            page_content=self.to_document_text(),
            metadata={
                "incident_id": self.incident_id,
                "location": self.incident_location,
                "neighborhood": self.neighborhood_name,
                "category": self.category,
                "priority": self.priority,
                "service_area": str(self.service_area),
                "response_time": self.total_response_time,
                "date": self.reported_at.isoformat(),
                "coordinates": f"{self.coordinates.latitude},{self.coordinates.longitude}",
                "postal_code": self.postal_code,
                "administrative_district": self.administrative_district
            }
        )


class QueryResult(BaseModel):
    """
    Structured container for query results including relevant incidents and metadata.
    """
    query: str = Field(..., description="Original query text")
    response_text: str = Field(..., description="Generated response")
    relevant_incidents: List[Incident] = Field(default_factory=list, description="Matching incidents")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional result metadata")
    confidence_score: Optional[float] = Field(None, description="Confidence score for the result")


class IncidentStatistics(BaseModel):
    """
    Comprehensive statistical summary of incident data for reporting and analysis.
    """
    total_incidents: int = Field(..., description="Total number of incidents")
    avg_response_time: float = Field(..., description="Average response time in minutes")
    avg_total_time: float = Field(..., description="Average total incident time in minutes")
    priority_distribution: Dict[str, int] = Field(default_factory=dict, description="Count by priority level")
    category_distribution: Dict[str, int] = Field(default_factory=dict, description="Count by category")
    service_area_distribution: Dict[int, int] = Field(default_factory=dict, description="Count by service area")
    response_time_categories: Dict[str, int] = Field(default_factory=dict, description="Count by response performance")