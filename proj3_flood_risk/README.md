# Detroit Flood Risk Hotspot Analysis Tool

A comprehensive Python-based analysis tool for identifying and visualizing flood risk hotspots in Detroit using multiple data sources including precipitation patterns, FEMA claims, impervious surface data, and infrastructure information.

## 🌊 Project Overview

This tool implements a sophisticated flood risk assessment methodology that combines:
- **Impervious Surface Analysis** (30% weight)
- **FEMA Claims Density** (25% weight)  
- **Precipitation Vulnerability** (20% weight)
- **Combined Sewer Overflow (CSO) Proximity** (15% weight)
- **Service Request Frequency** (10% weight)

The analysis generates interactive maps, dashboards, and detailed reports to help identify areas most vulnerable to flooding during rain events.

## 📊 Data Sources

### CSV Files
- `Detroit Censust Tracts Impervious Percent.csv` - Census tract impervious surface percentages (2015-2023)
- `Detroit Neighborhoods Impervious Percent.csv` - Neighborhood-level impervious surface data
- `FemaNfipRedactedClaims_DetroitZip.csv` - 806 FEMA flood insurance claims with damage amounts
- `NRI_Table_CensusTracts_Michigan.csv` - National Risk Index data with flood risk scores
- `Green Stormwater Infrastructure Locations.csv` - 299 green infrastructure projects
- `Improve_Detroit_Redacted.csv` - 128,454 service requests including flooding complaints
- `Precip_6hour_DetroitCityAirport_2015-2025.csv` - Hourly precipitation data (Detroit Airport)
- `Precip_6hour_DTW_2015-2025.csv` - Hourly precipitation data (DTW Airport)
- `Precip_DailyInches_WayneCountyStations_2015-2025.csv` - Daily precipitation data

### GeoJSON Files
- `FEMA_Flood_Hazard_Areas.geojson` - FEMA designated flood zones
- `combined_sewer_area.geojson` - Combined sewer overflow areas
- `Uncontrolled_CSO_Outfalls_to_the_Rouge_River.geojson` - 45 CSO outfall points
- `Detroit_River_Watershed.geojson` - Watershed boundaries
- Infrastructure files (roads, bridges, culverts, pump stations with risk ratings)

## 🔧 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- Git (for cloning the repository)

### Required Packages
Install the required Python packages using pip:

```bash
pip install pandas geopandas numpy matplotlib seaborn folium plotly warnings datetime
```

Or install all at once:
```bash
pip install pandas geopandas numpy matplotlib seaborn folium plotly
```

### Clone Repository
```bash
git clone https://github.com/Chudbrochil/MIDAS-Hackathon-Aug2025.git
cd MIDAS-Hackathon-Aug2025/proj3_flood_risk
```

## 🚀 How to Run

### Quick Start
1. **Ensure all data files are in the correct directory:**
   ```
   proj3_flood_risk/
   ├── flood_risk_hotspot_analysis.py
   └── Project 3_ Flood and Erosion Risk Policy Analysis Tool/
       ├── Detroit Censust Tracts Impervious Percent.csv
       ├── FemaNfipRedactedClaims_DetroitZip.csv
       ├── [all other data files...]
   ```

2. **Run the analysis:**
   ```bash
   python flood_risk_hotspot_analysis.py
   ```

3. **Expected output:**
   ```
   Loading data files...
   Data loading completed!
   Preprocessing data...
   Data preprocessing completed!
   Calculating hotspot scores...
   Hotspot calculation completed! 282 areas analyzed.
   Creating flood hotspot map...
   Flood hotspot map created successfully!
   Interactive map saved as detroit_flood_hotspots_map.html
   Analyzing precipitation-flood correlation...
   Creating interactive dashboard...
   Dashboard saved as flood_hotspot_dashboard.html
   Generating comprehensive report...
   ```

### Custom Data Path
If your data is in a different location, modify the `data_path` variable in the `main()` function:

```python
# Update this path to your data directory
data_path = "your/custom/path/to/data"
```

## 📈 Outputs Generated

The analysis generates the following files:

### 1. Interactive Map (`detroit_flood_hotspots_map.html`)
- **Color-coded risk levels:** Low (green), Moderate (orange), High (red), Very High (purple)
- **Detailed popups** with component scores for each census tract
- **CSO outfall locations** marked with blue icons
- **Multiple map layers** and interactive controls

### 2. Interactive Dashboard (`flood_hotspot_dashboard.html`)
- **Hotspot Score Distribution** histogram
- **Risk Level by Component** bar charts
- **Impervious Surface vs Flood Risk** scatter plot
- **Geographic Risk Distribution** summary

### 3. Detailed Scores (`flood_hotspot_scores.csv`)
- Complete scoring breakdown for each census tract
- Component scores and total risk assessment
- Risk level classifications

### 4. Correlation Analysis (`precipitation_correlation_analysis.png`)
- **Seasonal patterns** of extreme precipitation events
- **FEMA claims timing** correlation
- **Temporal trends** (2015-2025)

### 5. Console Report
Comprehensive text summary including:
- Total areas analyzed
- Risk level distribution
- Average component scores
- Top 10 highest risk areas

## 🎯 Hotspot Scoring Methodology

### Scoring Algorithm
```python
Total_Score = (Impervious_Surface_Pct × 0.3) + 
              (FEMA_Claims_Density × 0.25) + 
              (Precipitation_Vulnerability × 0.2) + 
              (CSO_Proximity × 0.15) + 
              (Service_Request_Frequency × 0.1)
```

### Risk Level Classification
- **Low Risk:** 0-25 points
- **Moderate Risk:** 25-50 points  
- **High Risk:** 50-75 points
- **Very High Risk:** 75-100 points

### Key Assumptions
1. **Impervious Surface Threshold:** Areas with >60% impervious surfaces are high-risk
2. **Precipitation Trigger:** 6-hour rainfall events >2 inches likely cause flooding
3. **Historical Pattern Weight:** Areas with 3+ FEMA claims indicate systemic issues
4. **Infrastructure Buffer:** 500m radius around CSO outfalls represents immediate risk
5. **Temporal Focus:** 2020-2025 data weighted more heavily for current conditions

## 📋 Analysis Features

### Core Components
- **Data Integration:** Combines multiple flood risk indicators
- **Spatial Analysis:** Geographic correlation of risk factors
- **Temporal Analysis:** Historical trends and seasonal patterns
- **Interactive Visualization:** Maps and dashboards for exploration
- **Statistical Correlation:** Precipitation-flood claim relationships

### Next Steps Implementation
1. **Precipitation Correlation:** Extreme events (>2" in 6 hours) vs FEMA claims
2. **Spatial Overlays:** Impervious surfaces, CSO locations, flood zones
3. **Service Request Patterns:** Flooding complaints during rain events
4. **Recent Data Weighting:** 2020-2025 emphasized for current assessment

## 🔍 Troubleshooting

### Common Issues

**File Not Found Errors:**
```bash
# Ensure data files are in correct directory structure
ls "Project 3_ Flood and Erosion Risk Policy Analysis Tool/"
```

**Large File Handling:**
- Some GeoJSON files (like FEMA flood zones) may be >50MB
- The script handles this gracefully with fallback approaches

**Memory Issues:**
- If running into memory constraints, consider processing subsets of data
- Large precipitation datasets are processed efficiently

**Missing Dependencies:**
```bash
# Install missing packages
pip install [package_name]
```

## 📝 Example Usage

```python
from flood_risk_hotspot_analysis import FloodHotspotAnalyzer

# Initialize analyzer
analyzer = FloodHotspotAnalyzer("path/to/data")

# Run complete analysis
analyzer.load_data()
analyzer.preprocess_data()
hotspot_scores = analyzer.calculate_hotspot_scores()
flood_map = analyzer.create_flood_map()
dashboard = analyzer.create_dashboard()
report = analyzer.generate_report()

# Access results
high_risk_areas = hotspot_scores[hotspot_scores['risk_level'] == 'Very High']
print(f"Found {len(high_risk_areas)} very high risk areas")
```

## 🤝 Contributing

This project was developed for the MIDAS Hackathon (August 2025). 

### Development Team
- **Repository:** [MIDAS-Hackathon-Aug2025](https://github.com/Chudbrochil/MIDAS-Hackathon-Aug2025)
- **Branch:** `prhanuma/project3-MIDAS`
- **Project:** Project 3 - Flood and Erosion Risk Policy Analysis Tool

### Future Enhancements
- Real-time precipitation data integration
- Machine learning risk prediction models
- Mobile-responsive dashboard
- Policy recommendation engine
- Climate change projection modeling

## 📄 License

This project is part of the MIDAS Hackathon 2025. Please refer to the repository license for usage terms.

## 📞 Support

For questions or issues:
1. Check the troubleshooting section above
2. Review the console output for detailed error messages
3. Ensure all data files are properly formatted and accessible
4. Verify Python package installations

---

**Last Updated:** August 1, 2025  
**Version:** 1.0.0  
**Python Version:** 3.8+