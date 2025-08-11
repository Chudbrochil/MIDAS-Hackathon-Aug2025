"""
A comprehensive web application for analyzing emergency response data in Detroit.
Provides interactive dashboards, natural language querying, and advanced analytics
for incident management and performance monitoring.
"""

import json
import logging
import random
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List

import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image

# Local application imports
from src.data_manager import DataManager
from src.data_models import IncidentPriority
from src.query_processor import QueryProcessor

# Configure application logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Streamlit page configuration
st.set_page_config(
    page_title="Detroit Open Data Portal",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 600;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stAlert {
        margin-top: 1rem;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    .success-metric {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 10px;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def initialize_application_systems():
    """
    Initialize the core application systems including data manager and query processor.

    This function handles system initialization with proper error handling and logging.
    Uses Streamlit's caching to ensure systems are initialized only once per session.

    Returns:
        tuple: (DataManager, QueryProcessor) instances or (None, None) on failure
    """
    try:
        logger.info("Initializing Detroit Open Data Portal systems...")

        # Initialize the data management system
        data_manager = DataManager()
        logger.info("Data manager initialized successfully")

        # Attempt to load default incident data
        default_data_path = "**Add path to your data file**"
        try:
            incident_count = data_manager.load_from_geojson_file(default_data_path)
            logger.info(f"Successfully loaded {incident_count} incidents from {default_data_path}")
        except FileNotFoundError:
            logger.warning(f"Default data file not found at {default_data_path}")
        except Exception as load_error:
            logger.warning(f"Could not load default data: {load_error}")

        # Initialize the query processing system
        query_processor = QueryProcessor(data_manager)
        logger.info("Query processor initialized successfully")

        return data_manager, query_processor

    except Exception as system_error:
        logger.error(f"Failed to initialize application systems: {system_error}")
        st.error(f"System initialization failed: {system_error}")
        return None, None


@st.cache_data
def generate_demonstration_data() -> Dict[str, Any]:
    """
    Generate realistic demonstration data for system testing and presentations.

    Creates a comprehensive dataset with varied incident types, priorities, and timing
    patterns that reflect real emergency response scenarios in Detroit.

    Returns:
        dict: GeoJSON formatted data structure with incident features
    """
    logger.info("Generating demonstration incident data...")

    # Define realistic incident categories
    incident_categories = [
        "Medical Emergency", "Fire", "Traffic Accident", "Utility Outage",
        "Security Incident", "Environmental Hazard", "Infrastructure Failure"
    ]

    # Detroit-area neighborhoods for realistic geographic distribution
    detroit_neighborhoods = [
        "Downtown", "Riverside", "Hillcrest", "Industrial District",
        "Residential North", "Commercial Center", "University Area"
    ]

    # Available priority levels
    available_priorities = list(IncidentPriority)

    # Generate incident records
    incident_records = []
    base_date = datetime.now() - timedelta(days=90)

    for incident_index in range(500):
        # Generate realistic timing with weighted priority distribution
        incident_timestamp = base_date + timedelta(days=random.randint(0, 90))

        # Create realistic priority distribution (not uniform)
        # Critical: 10%, High: 25%, Medium: 45%, Low: 20%
        priority_weights = [0.10, 0.25, 0.45, 0.20]
        selected_priority = random.choices(available_priorities, weights=priority_weights)[0]

        # Generate response times based on priority level for realism
        if selected_priority == IncidentPriority.CRITICAL:
            base_response = random.uniform(3, 12)  # Critical incidents get fastest response
            urgency_description = "Emergency response required"
        elif selected_priority == IncidentPriority.HIGH:
            base_response = random.uniform(6, 18)  # High priority incidents
            urgency_description = "Urgent response required"
        elif selected_priority == IncidentPriority.MEDIUM:
            base_response = random.uniform(10, 25)  # Medium priority incidents
            urgency_description = "Standard response required"
        else:  # LOW priority
            base_response = random.uniform(15, 35)  # Low priority incidents
            urgency_description = "Routine service call"

        # Add some natural variation to response times
        response_duration = max(2, base_response + random.gauss(0, 3))

        # Select category that might correlate with priority
        if selected_priority in [IncidentPriority.CRITICAL, IncidentPriority.HIGH]:
            # Higher chance of medical emergencies and fires for high priority
            high_priority_categories = ["Medical Emergency", "Fire", "Traffic Accident"]
            if random.random() < 0.7:  # 70% chance
                selected_category = random.choice(high_priority_categories)
            else:
                selected_category = random.choice(incident_categories)
        else:
            selected_category = random.choice(incident_categories)

        # Create incident data structure
        incident_record = {
            "incident_entry_id": incident_index + 1,
            "incident_id": f"INC-2024-{incident_index + 1:04d}",
            "incident_location": f"{random.randint(100, 9999)} {random.choice(['Main St', 'Oak Ave', 'Pine Rd', 'Cedar Blvd'])}",
            "coordinates": {
                "longitude": round(random.uniform(-83.3, -83.0), 6),
                "latitude": round(random.uniform(42.2, 42.5), 6)
            },
            "postal_code": f"{random.randint(48000, 48999)}",
            "service_area": random.randint(1, 8),
            "district": f"District-{random.randint(1, 12)}",
            "neighborhood_name": random.choice(detroit_neighborhoods),
            "administrative_district": random.randint(1, 7),
            "incident_source": random.choice([
                "Emergency Line", "Walk-in", "Radio", "Online", "Mobile App"
            ]),
            "incident_description": f"{selected_category} - {urgency_description}",
            "category": selected_category,
            "priority": selected_priority.value,  # Use .value to get string representation
            "incident_type": f"Type-{random.randint(1, 20)}",
            "incident_code": f"CODE-{random.randint(100, 999)}",
            "reported_at": incident_timestamp.isoformat(),
            "intake_time": round(random.uniform(1, 5), 2),
            "dispatch_time": round(random.uniform(2, 8), 2),
            "travel_time": round(random.uniform(5, 25), 2),
            "on_scene_time": round(random.uniform(10, 60), 2),
            "total_response_time": round(response_duration, 2),
            "total_time": round(response_duration + random.uniform(20, 120), 2)
        }
        incident_records.append(incident_record)

    # Structure data as GeoJSON for geographic compatibility
    geojson_structure = {
        "type": "FeatureCollection",
        "features": []
    }

    for incident in incident_records:
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [
                    incident["coordinates"]["longitude"],
                    incident["coordinates"]["latitude"]
                ]
            },
            "properties": incident
        }
        geojson_structure["features"].append(feature)

    logger.info(f"Generated {len(incident_records)} demonstration incidents")
    return geojson_structure


def display_key_performance_metrics(statistics) -> None:
    """
    Display essential performance metrics in a professional dashboard layout.

    Args:
        statistics: IncidentStatistics object containing aggregated metrics
    """
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

    with metric_col1:
        st.metric(
            label="📊 Total Incidents",
            value=f"{statistics.total_incidents:,}",
            help="Total number of incidents in the database"
        )

    with metric_col2:
        st.metric(
            label="⏱️ Avg Response Time",
            value=f"{statistics.avg_response_time:.1f} min",
            help="Average time from incident report to first responder arrival"
        )

    with metric_col3:
        st.metric(
            label="🕐 Avg Total Time",
            value=f"{statistics.avg_total_time:.1f} min",
            help="Average total time from report to incident closure"
        )

    with metric_col4:
        high_priority_incidents = (
                statistics.priority_distribution.get('Critical', 0) +
                statistics.priority_distribution.get('High', 0)
        )
        st.metric(
            label="🚨 High Priority",
            value=f"{high_priority_incidents:,}",
            help="Number of Critical and High priority incidents"
        )


def create_priority_distribution_visualization(statistics):
    """
    Generate a pie chart showing the distribution of incident priorities.

    Args:
        statistics: IncidentStatistics object with priority distribution data

    Returns:
        plotly.graph_objects.Figure: Pie chart visualization
    """
    priority_labels = list(statistics.priority_distribution.keys())
    priority_counts = list(statistics.priority_distribution.values())

    visualization = px.pie(
        values=priority_counts,
        names=priority_labels,
        title="Incident Priority Distribution",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    visualization.update_traces(textposition='inside', textinfo='percent+label')
    return visualization


def create_category_distribution_visualization(statistics):
    """
    Generate a horizontal bar chart for incident category distribution.

    Args:
        statistics: IncidentStatistics object with category distribution data

    Returns:
        plotly.graph_objects.Figure: Horizontal bar chart visualization
    """
    # Display top 10 categories for readability
    top_categories = list(statistics.category_distribution.keys())[:10]
    category_counts = [statistics.category_distribution[cat] for cat in top_categories]

    visualization = px.bar(
        x=category_counts,
        y=top_categories,
        orientation='h',
        title="Top 10 Incident Categories",
        labels={'x': 'Number of Incidents', 'y': 'Category'}
    )
    visualization.update_layout(height=400)
    return visualization


def create_response_performance_visualization(statistics):
    """
    Generate a color-coded bar chart showing response time performance categories.

    Args:
        statistics: IncidentStatistics object with response time category data

    Returns:
        plotly.graph_objects.Figure: Color-coded bar chart
    """
    performance_categories = list(statistics.response_time_categories.keys())
    performance_counts = list(statistics.response_time_categories.values())

    # Define performance-based color scheme
    performance_colors = {
        'Excellent': 'green',
        'Good': 'lightgreen',
        'Fair': 'orange',
        'Poor': 'red'
    }
    color_mapping = [performance_colors.get(cat, 'blue') for cat in performance_categories]

    visualization = px.bar(
        x=performance_categories,
        y=performance_counts,
        title="Response Time Performance",
        labels={'x': 'Response Category', 'y': 'Number of Incidents'},
        color=performance_categories,
        color_discrete_map=performance_colors
    )
    return visualization


def create_temporal_trend_visualization(dataframe):
    """
    Generate a time series line chart showing daily incident volume trends.

    Args:
        dataframe: pandas DataFrame with incident data including 'reported_at' column

    Returns:
        plotly.graph_objects.Figure or None: Time series visualization
    """
    if dataframe.empty:
        return None

    # Prepare daily aggregation
    dataframe_copy = dataframe.copy()
    dataframe_copy['date'] = pd.to_datetime(dataframe_copy['reported_at']).dt.date
    daily_aggregation = dataframe_copy.groupby('date').size().reset_index(name='count')

    visualization = px.line(
        daily_aggregation,
        x='date',
        y='count',
        title="Daily Incident Volume",
        labels={'date': 'Date', 'count': 'Number of Incidents'}
    )
    visualization.update_layout(height=400)
    return visualization


def create_geographic_incident_map(dataframe):
    """
    Generate an interactive map visualization of incident locations.

    Args:
        dataframe: pandas DataFrame with geographic and incident data

    Returns:
        plotly.graph_objects.Figure or None: Interactive map visualization
    """
    if dataframe.empty:
        return None

    visualization = px.scatter_mapbox(
        dataframe,
        lat='latitude',
        lon='longitude',
        color='priority',
        size='total_response_time',
        hover_data=['incident_id', 'category', 'neighborhood_name'],
        mapbox_style='open-street-map',
        title="Incident Locations",
        height=600,
        zoom=10
    )

    # Center the map on the geographic center of the data
    visualization.update_layout(
        mapbox=dict(
            center=dict(
                lat=dataframe['latitude'].mean(),
                lon=dataframe['longitude'].mean()
            )
        )
    )
    return visualization


def main():
    """
    Main application entry point that handles page routing and core functionality.
    """
    # Application header with branding
    try:
        logo_image = Image.open("Detroit_Open_Portal_Logo.png")
        header_col1, header_col2 = st.columns([1, 8])

        with header_col1:
            st.image(logo_image, width=50)

        with header_col2:
            st.markdown("""
                <div style="display: flex; align-items: center; height: 100%;">
                    <h1 style='margin-bottom: 0;'>City of Detroit Open Data Portal</h1>
                </div>
            """, unsafe_allow_html=True)
    except FileNotFoundError:
        st.markdown("# 🏛️ City of Detroit Open Data Portal")
        logger.warning("Logo file not found, using text header")

    # Initialize core application systems
    data_manager, query_processor = initialize_application_systems()

    if data_manager is None or query_processor is None:
        st.error("⚠️ Failed to initialize the system. Please check your configuration.")
        st.info("💡 Ensure all required dependencies are installed and configured properly.")
        return

    # Navigation sidebar
    st.sidebar.title("📋 Navigation")
    selected_page = st.sidebar.selectbox(
        "Choose a page",
        ["Dashboard", "Query Interface", "Data Upload", "Analytics"],
        help="Select a section to explore different features"
    )

    # Route to appropriate page based on user selection
    if selected_page == "Dashboard":
        render_dashboard_page(data_manager, query_processor)
    elif selected_page == "Query Interface":
        render_query_interface_page(data_manager, query_processor)
    elif selected_page == "Data Upload":
        render_data_upload_page(data_manager)
    elif selected_page == "Analytics":
        render_analytics_page(data_manager)


def render_dashboard_page(data_manager, query_processor) -> None:
    """
    Render the main dashboard page with key metrics and visualizations.

    Args:
        data_manager: DataManager instance for accessing incident data
        query_processor: QueryProcessor instance for data analysis
    """
    st.header("📊 Emergency Response Dashboard")

    # Check data availability
    if len(data_manager.incidents) == 0:
        st.warning("⚠️ No incident data is currently loaded.")
        st.info("💡 Load demonstration data to explore the portal's capabilities.")

        if st.button("📥 Load Sample Data", type="primary"):
            with st.spinner("Generating and loading demonstration data..."):
                demonstration_data = generate_demonstration_data()
                loaded_count = data_manager.load_from_geojson(demonstration_data)
                st.success(f"✅ Successfully loaded {loaded_count} demonstration incidents!")
                st.rerun()
        return

    # Generate and display key statistics
    incident_statistics = data_manager.calculate_statistics()
    display_key_performance_metrics(incident_statistics)

    # Create visualization layout
    visualization_col1, visualization_col2 = st.columns(2)

    with visualization_col1:
        # Priority distribution visualization
        priority_chart = create_priority_distribution_visualization(incident_statistics)
        st.plotly_chart(priority_chart, use_container_width=True)

        # Response time performance visualization
        response_performance_chart = create_response_performance_visualization(incident_statistics)
        st.plotly_chart(response_performance_chart, use_container_width=True)

    with visualization_col2:
        # Category distribution visualization
        category_chart = create_category_distribution_visualization(incident_statistics)
        st.plotly_chart(category_chart, use_container_width=True)

        # Temporal trend visualization
        incident_dataframe = data_manager.to_dataframe()
        trend_chart = create_temporal_trend_visualization(incident_dataframe)
        if trend_chart:
            st.plotly_chart(trend_chart, use_container_width=True)

    # Geographic visualization section
    st.subheader("🗺️ Geographic Distribution")
    geographic_map = create_geographic_incident_map(incident_dataframe)
    if geographic_map:
        st.plotly_chart(geographic_map, use_container_width=True)
    else:
        st.info("📍 Geographic visualization requires location data.")


def render_query_interface_page(data_manager, query_processor) -> None:
    """
    Render the natural language query interface page.

    Args:
        data_manager: DataManager instance for data access
        query_processor: QueryProcessor instance for query processing
    """
    st.header("💬 Natural Language Query Interface")

    if len(data_manager.incidents) == 0:
        st.warning("⚠️ No incident data available for querying. Please load data first.")
        return

    # Sample query suggestions
    st.subheader("💡 Sample Queries")
    suggested_queries = [
        "What are the average response times by neighborhood?",
        "Which service areas have the highest incident volume?",
        "Show me critical incidents from the last week",
        "What are the most common incident categories?",
        "Which neighborhoods have the longest response times?",
        "How do response times vary by priority level?",
        "What's the trend in incident volume over time?"
    ]

    selected_suggestion = st.selectbox(
        "Select a sample query:",
        [""] + suggested_queries,
        help="Choose a pre-defined query or write your own"
    )

    # Query input interface
    user_query = st.text_area(
        "Enter your query:",
        value=selected_suggestion,
        height=100,
        placeholder="Ask about incident patterns, response times, locations, etc."
    )

    if st.button("🔍 Submit Query", type="primary"):
        if user_query.strip():
            with st.spinner("Processing your query..."):
                try:
                    query_results = query_processor.process_query(user_query)

                    # Display query response
                    st.subheader("📋 Analysis Results")
                    st.write(query_results["text"])

                    # Display relevant incidents
                    if query_results["relevant_incidents"]:
                        st.subheader(f"📄 Relevant Incidents ({len(query_results['relevant_incidents'])})")

                        # Convert incidents to display format
                        incident_display_data = []
                        for incident in query_results["relevant_incidents"][:10]:  # Show top 10
                            incident_display_data.append({
                                "ID": incident.incident_id,
                                "Location": incident.incident_location,
                                "Category": incident.category,
                                "Priority": incident.priority,
                                "Response Time": f"{incident.total_response_time:.1f} min",
                                "Date": incident.reported_at.strftime("%Y-%m-%d %H:%M")
                            })

                        st.dataframe(
                            pd.DataFrame(incident_display_data),
                            use_container_width=True
                        )

                    # Visualization of results
                    if query_results["data"] is not None and not query_results["data"].empty:
                        st.subheader("📊 Data Visualization")

                        result_visualization = px.scatter(
                            query_results["data"],
                            x="reported_at",
                            y="total_response_time",
                            color="priority",
                            title="Response Times for Relevant Incidents",
                            labels={
                                "reported_at": "Date",
                                "total_response_time": "Response Time (min)"
                            }
                        )
                        st.plotly_chart(result_visualization, use_container_width=True)

                except Exception as query_error:
                    st.error(f"❌ Error processing query: {query_error}")
                    logger.error(f"Query processing error: {query_error}")
        else:
            st.warning("⚠️ Please enter a query to analyze.")


def render_data_upload_page(data_manager) -> None:
    """
    Render the data upload and management page.

    Args:
        data_manager: DataManager instance for data operations
    """
    st.header("📁 Data Upload and Management")

    st.markdown("Upload your incident data in GeoJSON format for analysis.")

    # File upload interface
    uploaded_file = st.file_uploader(
        "Choose a GeoJSON file",
        type=['json', 'geojson'],
        help="Upload a GeoJSON file containing incident data"
    )

    if uploaded_file is not None:
        try:
            # Read and validate uploaded file
            uploaded_data = json.load(uploaded_file)

            # Display file preview
            st.subheader("📋 File Preview")
            if "features" in uploaded_data:
                feature_count = len(uploaded_data['features'])
                st.success(f"✅ Valid GeoJSON file with {feature_count} incident features")

                # Show sample feature structure
                if uploaded_data["features"]:
                    with st.expander("👁️ Sample Data Structure"):
                        st.json(uploaded_data["features"][0], expanded=False)

                if st.button("📥 Load Data", type="primary"):
                    with st.spinner("Loading incident data..."):
                        loaded_count = data_manager.load_from_geojson(uploaded_data)
                        st.success(f"🎉 Successfully loaded {loaded_count} incidents!")

        except json.JSONDecodeError:
            st.error("❌ Invalid JSON format. Please check your file.")
        except Exception as file_error:
            st.error(f"❌ Error reading file: {file_error}")

    # Current data status display
    st.subheader("📊 Current Data Status")
    current_incident_count = len(data_manager.incidents)
    if current_incident_count > 0:
        st.info(f"📈 Currently loaded: {current_incident_count:,} incidents")
    else:
        st.info("📭 No incidents currently loaded")

    # Data management actions
    if current_incident_count > 0:
        if st.button("🗑️ Clear All Data", type="secondary"):
            st.warning("⚠️ Data clearing functionality not implemented in this version")


def render_analytics_page(data_manager) -> None:
    """
    Render the advanced analytics page with filtering and detailed analysis.

    Args:
        data_manager: DataManager instance for data access
    """
    st.header("📈 Advanced Analytics")

    if len(data_manager.incidents) == 0:
        st.warning("⚠️ No incident data available for analysis. Please load data first.")
        return

    incident_dataframe = data_manager.to_dataframe()

    # Data filtering controls
    st.subheader("🔍 Data Filters")
    filter_col1, filter_col2, filter_col3 = st.columns(3)

    with filter_col1:
        selected_priorities = st.multiselect(
            "Priority Levels",
            options=incident_dataframe['priority'].unique(),
            default=incident_dataframe['priority'].unique(),
            help="Filter incidents by priority level"
        )

    with filter_col2:
        selected_categories = st.multiselect(
            "Incident Categories",
            options=incident_dataframe['category'].unique(),
            default=incident_dataframe['category'].unique()[:5],  # Default to first 5
            help="Filter incidents by category type"
        )

    with filter_col3:
        selected_service_areas = st.multiselect(
            "Service Areas",
            options=sorted(incident_dataframe['service_area'].unique()),
            default=sorted(incident_dataframe['service_area'].unique())[:3],  # Default to first 3
            help="Filter incidents by service area"
        )

    # Apply selected filters
    filtered_dataframe = incident_dataframe[
        (incident_dataframe['priority'].isin(selected_priorities)) &
        (incident_dataframe['category'].isin(selected_categories)) &
        (incident_dataframe['service_area'].isin(selected_service_areas))
        ]

    st.info(f"📊 Displaying {len(filtered_dataframe)} incidents (filtered from {len(incident_dataframe)} total)")

    # Advanced analytics visualizations
    if not filtered_dataframe.empty:
        # Response time analysis by neighborhood
        st.subheader("⏱️ Response Time Analysis")

        neighborhood_response_times = (
            filtered_dataframe.groupby('neighborhood_name')['total_response_time']
            .mean()
            .sort_values(ascending=False)
        )

        neighborhood_analysis_chart = px.bar(
            x=neighborhood_response_times.values,
            y=neighborhood_response_times.index,
            orientation='h',
            title="Average Response Time by Neighborhood",
            labels={'x': 'Average Response Time (minutes)', 'y': 'Neighborhood'}
        )
        st.plotly_chart(neighborhood_analysis_chart, use_container_width=True)

        # Temporal pattern analysis
        st.subheader("📅 Incident Patterns")

        # Prepare temporal data
        filtered_dataframe_copy = filtered_dataframe.copy()
        filtered_dataframe_copy['hour'] = pd.to_datetime(filtered_dataframe_copy['reported_at']).dt.hour
        filtered_dataframe_copy['day_name'] = pd.to_datetime(filtered_dataframe_copy['reported_at']).dt.day_name()

        # Create incident heatmap
        heatmap_data = filtered_dataframe_copy.groupby(['day_name', 'hour']).size().unstack(fill_value=0)

        # Reorder days for logical display
        day_ordering = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        heatmap_data = heatmap_data.reindex(day_ordering)

        heatmap_visualization = px.imshow(
            heatmap_data.values,
            x=heatmap_data.columns,
            y=heatmap_data.index,
            title="Incident Heatmap (Hour vs Day of Week)",
            labels={'x': 'Hour of Day', 'y': 'Day of Week', 'color': 'Number of Incidents'},
            aspect='auto'
        )
        st.plotly_chart(heatmap_visualization, use_container_width=True)

    else:
        st.warning("🔍 No incidents match the selected filters. Please adjust your criteria.")


if __name__ == "__main__":
    main()