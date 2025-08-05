"""
Synthetic Blight Dataset Generator
=================================

This script generates realistic synthetic datasets for blight detection modeling.
Creates two datasets: small (1,000 samples) and large (100,000 samples) with
realistic parcel features and correlated blight classifications.

Blight Classification Scale:
- 1: No blight (well-maintained property)
- 2: Minor blight (cosmetic issues, minor repairs needed)  
- 3: Moderate blight (structural issues, significant deterioration)
- 4: Severe blight (unsafe/uninhabitable, potential demolition)

Output: CSV files with parcel data ready for machine learning
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

def generate_synthetic_dataset(n_samples, dataset_name="dataset"):
    """
    Generate synthetic parcel data for blight detection.
    
    Features are based on realistic data sources:
    - Property records (age, size, value)
    - Tax assessment data (delinquency, assessed value)
    - Code enforcement (violations, inspections)
    - USPS data (mail holds indicating vacancy)
    - Neighborhood demographics and crime data
    - Infrastructure and utility data
    
    Args:
        n_samples (int): Number of samples to generate
        dataset_name (str): Name identifier for the dataset
        
    Returns:
        pd.DataFrame: Generated synthetic dataset with realistic correlations
    """
    print(f"Generating {dataset_name} with {n_samples:,} samples...")
    
    # Generate parcel IDs (realistic format)
    parcel_ids = [f"PARL{str(i).zfill(8)}" for i in range(1, n_samples + 1)]
    
    # Generate realistic feature distributions based on real-world data
    data = {
        'parcel_id': parcel_ids,
        
        # Property characteristics
        'property_age': np.random.gamma(2, 15, n_samples).astype(int),  # Age in years, skewed toward older
        'square_footage': np.random.lognormal(7.5, 0.5, n_samples).astype(int),  # Log-normal distribution
        'assessed_value': np.random.lognormal(11, 0.8, n_samples).astype(int),  # Property value in dollars
        'lot_size': np.random.gamma(2, 2000, n_samples).astype(int),  # Lot size in sq ft
        
        # Financial indicators
        'tax_delinquent_years': np.random.poisson(0.8, n_samples),  # Years behind on taxes
        'tax_amount_owed': np.random.exponential(2000, n_samples).astype(int),  # Tax debt amount
        
        # Code enforcement and inspections
        'num_code_violations': np.random.poisson(1.2, n_samples),  # Violations in last 3 years
        'days_since_last_inspection': np.random.exponential(400, n_samples).astype(int),  # Days since inspection
        'open_violations': np.random.poisson(0.6, n_samples),  # Currently open violations
        
        # USPS and vacancy indicators
        'vacant_mail_holds': np.random.poisson(0.3, n_samples),  # Mail holds indicating vacancy
        'delivery_issues': np.random.poisson(0.5, n_samples),  # Mail delivery problems
        
        # Neighborhood context
        'neighborhood_median_income': np.random.normal(45000, 15000, n_samples).astype(int),
        'crime_incidents_nearby': np.random.poisson(2.5, n_samples),  # Crimes within 500ft, 1 year
        'distance_to_downtown': np.random.exponential(8, n_samples),  # Miles to city center
        'num_foreclosures_nearby': np.random.poisson(0.7, n_samples),  # Foreclosures within 0.25mi, 2 years
        'vacant_lots_nearby': np.random.poisson(1.8, n_samples),  # Vacant lots within 500ft
        
        # Infrastructure and utilities
        'utility_shutoffs': np.random.poisson(0.4, n_samples),  # Gas/electric disconnections
        'water_shutoffs': np.random.poisson(0.3, n_samples),  # Water service disconnections
        'sidewalk_condition': np.random.choice([1, 2, 3, 4, 5], n_samples, 
                                             p=[0.1, 0.2, 0.4, 0.2, 0.1]),  # 1=poor, 5=excellent
        
        # Property condition indicators  
        'building_material_quality': np.random.choice([1, 2, 3, 4, 5], n_samples, 
                                                    p=[0.1, 0.2, 0.4, 0.2, 0.1]),  # Construction quality
        'roof_condition': np.random.choice([1, 2, 3, 4, 5], n_samples,
                                         p=[0.15, 0.25, 0.3, 0.2, 0.1]),  # Roof condition
        'lot_condition': np.random.choice([1, 2, 3, 4, 5], n_samples,
                                        p=[0.15, 0.25, 0.3, 0.2, 0.1])  # Lot maintenance
    }
    
    df = pd.DataFrame(data)
    
    # Apply realistic constraints and correlations
    df.loc[df['property_age'] > 150, 'property_age'] = 150  # Cap at 150 years
    df.loc[df['property_age'] < 0, 'property_age'] = 0
    df.loc[df['assessed_value'] < 5000, 'assessed_value'] = 5000  # Minimum property value
    df.loc[df['neighborhood_median_income'] < 15000, 'neighborhood_median_income'] = 15000
    df.loc[df['tax_delinquent_years'] > 15, 'tax_delinquent_years'] = 15
    df.loc[df['days_since_last_inspection'] > 3000, 'days_since_last_inspection'] = 3000
    df.loc[df['lot_size'] < 1000, 'lot_size'] = 1000  # Minimum lot size
    
    # Create realistic correlations between features
    # Older properties tend to have more violations
    age_effect = (df['property_age'] / 50).clip(0, 2)
    df['num_code_violations'] = (df['num_code_violations'] * age_effect).astype(int)
    
    # Lower income areas tend to have more crime and foreclosures
    income_effect = (60000 / df['neighborhood_median_income'].clip(20000, 100000))
    df['crime_incidents_nearby'] = (df['crime_incidents_nearby'] * income_effect).astype(int)
    df['num_foreclosures_nearby'] = (df['num_foreclosures_nearby'] * income_effect).astype(int)
    
    # Tax delinquency correlates with property value
    value_effect = (df['assessed_value'].median() / df['assessed_value']).clip(0.5, 3)
    df['tax_delinquent_years'] = (df['tax_delinquent_years'] * value_effect).astype(int)
    
    # Generate blight labels with realistic correlations to all features
    blight_classes = generate_realistic_blight_labels(df)
    df['blight_level'] = blight_classes
    
    print(f"Generated {dataset_name}:")
    print(f"  - Shape: {df.shape}")
    print(f"  - Features: {len([col for col in df.columns if col not in ['parcel_id', 'blight_level']])}")
    
    # Display blight distribution
    blight_counts = df['blight_level'].value_counts().sort_index()
    print(f"  - Blight distribution:")
    for level in [1, 2, 3, 4]:
        count = blight_counts.get(level, 0)
        pct = count / len(df) * 100
        print(f"    Level {level}: {count:,} ({pct:.1f}%)")
    
    return df

def generate_realistic_blight_labels(df):
    """
    Generate blight labels with realistic correlations to property features.
    Uses a weighted scoring system based on multiple risk factors.
    """
    n_samples = len(df)
    blight_classes = []
    
    for i in range(n_samples):
        # Calculate risk factors (normalized to 0-2 scale)
        risk_factors = {
            'age': min(df.loc[i, 'property_age'] / 50, 2),
            'violations': min(df.loc[i, 'num_code_violations'] / 3, 2),
            'tax_delinquency': min(df.loc[i, 'tax_delinquent_years'] / 3, 2),
            'inspection_gap': min(df.loc[i, 'days_since_last_inspection'] / 500, 2),
            'vacancy': min(df.loc[i, 'vacant_mail_holds'] * 2, 2),
            'utilities': min((df.loc[i, 'utility_shutoffs'] + df.loc[i, 'water_shutoffs']) / 2, 2),
            'neighborhood': max(0.2, 60000 / max(df.loc[i, 'neighborhood_median_income'], 20000)),
            'property_condition': max(0.2, 4 / df.loc[i, 'building_material_quality']),
            'lot_condition': max(0.2, 4 / df.loc[i, 'lot_condition']),
            'roof_condition': max(0.2, 4 / df.loc[i, 'roof_condition']),
            'crime': min(df.loc[i, 'crime_incidents_nearby'] / 5, 2),
            'foreclosures': min(df.loc[i, 'num_foreclosures_nearby'] / 3, 2)
        }
        
        # Weighted average of risk factors
        weights = {
            'age': 0.15, 'violations': 0.20, 'tax_delinquency': 0.15,
            'inspection_gap': 0.10, 'vacancy': 0.15, 'utilities': 0.10,
            'neighborhood': 0.05, 'property_condition': 0.05, 'lot_condition': 0.02,
            'roof_condition': 0.02, 'crime': 0.003, 'foreclosures': 0.003
        }
        
        risk_score = sum(risk_factors[factor] * weights[factor] for factor in risk_factors)
        
        # Convert risk score to probabilities for each blight level
        if risk_score < 0.6:
            probs = [0.70, 0.20, 0.08, 0.02]  # Low risk - mostly no blight
        elif risk_score < 1.0:
            probs = [0.45, 0.35, 0.15, 0.05]  # Medium-low risk
        elif risk_score < 1.4:
            probs = [0.25, 0.35, 0.30, 0.10]  # Medium-high risk
        else:
            probs = [0.10, 0.25, 0.40, 0.25]  # High risk - more severe blight
        
        # Sample blight level based on probabilities
        blight_level = np.random.choice([1, 2, 3, 4], p=probs)
        blight_classes.append(blight_level)
    
    return blight_classes

def save_dataset(df, filename, format='csv'):
    """Save dataset to file with specified format."""
    filepath = Path(filename)
    
    if format.lower() == 'csv':
        df.to_csv(filepath.with_suffix('.csv'), index=False)
        print(f"✅ Saved to {filepath.with_suffix('.csv')}")
    elif format.lower() == 'tsv':
        df.to_csv(filepath.with_suffix('.tsv'), sep='\t', index=False)
        print(f"✅ Saved to {filepath.with_suffix('.tsv')}")
    else:
        raise ValueError("Format must be 'csv' or 'tsv'")
    
    return filepath.with_suffix(f'.{format}')

def main():
    """Main function to generate datasets."""
    parser = argparse.ArgumentParser(description='Generate synthetic blight detection datasets')
    parser.add_argument('--small-size', type=int, default=1000, 
                       help='Size of small dataset (default: 1000)')
    parser.add_argument('--large-size', type=int, default=100000,
                       help='Size of large dataset (default: 100000)')
    parser.add_argument('--format', choices=['csv', 'tsv'], default='csv',
                       help='Output format (default: csv)')
    parser.add_argument('--output-dir', type=str, default='data',
                       help='Output directory (default: data)')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("🏠 SYNTHETIC BLIGHT DATASET GENERATOR")
    print("=" * 50)
    print(f"Random seed: {RANDOM_STATE}")
    print(f"Output format: {args.format.upper()}")
    print(f"Output directory: {output_dir}")
    
    # Generate small dataset
    print(f"\n📊 Generating datasets...")
    small_dataset = generate_synthetic_dataset(args.small_size, "Small Dataset")
    small_file = save_dataset(small_dataset, output_dir / 'blight_data_small', args.format)
    
    # Generate large dataset
    large_dataset = generate_synthetic_dataset(args.large_size, "Large Dataset")
    large_file = save_dataset(large_dataset, output_dir / 'blight_data_large', args.format)
    
    # Summary statistics
    print(f"\n📈 DATASET SUMMARY:")
    print(f"Small dataset: {len(small_dataset):,} samples, {len(small_dataset.columns):,} columns")
    print(f"Large dataset: {len(large_dataset):,} samples, {len(large_dataset.columns):,} columns")
    
    print(f"\n🎉 Data generation complete!")
    print(f"Files created:")
    print(f"  - {small_file}")
    print(f"  - {large_file}")
    
    print(f"\nFeature columns ({len([col for col in small_dataset.columns if col not in ['parcel_id', 'blight_level']])}):")
    feature_cols = [col for col in small_dataset.columns if col not in ['parcel_id', 'blight_level']]
    for i, col in enumerate(feature_cols, 1):
        print(f"  {i:2d}. {col}")

if __name__ == "__main__":
    main()