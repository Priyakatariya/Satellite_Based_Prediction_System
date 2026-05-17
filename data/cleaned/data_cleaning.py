import pandas as pd
import numpy as np
import os
import re

def standardize_column_names(df):
    """
    Standardize column names to snake_case.
    Example: 'CO(GT)' -> 'co_gt', 'PM2.5' -> 'pm2_5'
    """
    new_columns = []
    for col in df.columns:
        # Lowercase
        col = col.lower()
        # Replace non-alphanumeric (except underscores) with underscores
        col = re.sub(r'[^a-z0-9_]', '_', col)
        # Replace multiple underscores with a single one
        col = re.sub(r'_+', '_', col)
        # Strip leading/trailing underscores
        col = col.strip('_')
        new_columns.append(col)
    df.columns = new_columns
    return df

def clean_air_quality_data(input_path, output_path):
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    
    # 1. Standardize columns
    print("Standardizing column names...")
    df = standardize_column_names(df)
    
    # 2. Handle missing values
    # In some air quality datasets, -200 is used as a null value.
    # Let's replace -200 with NaN if it exists.
    df = df.replace(-200, np.nan)
    
    # 3. Date/Time parsing
    if 'date' in df.columns and 'time' in df.columns:
        print("Parsing date and time into timestamp...")
        # Assuming date format is YYYY-MM-DD and time is HH:MM
        df['timestamp'] = pd.to_datetime(df['date'] + ' ' + df['time'], errors='coerce')
        # Reorder columns to put timestamp at the beginning
        cols = ['timestamp'] + [c for col in df.columns if (c := col) != 'timestamp']
        df = df[cols]
    
    # 4. Save cleaned data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")
    print(f"Final columns: {list(df.columns)}")

if __name__ == "__main__":
    raw_path = os.path.join('data', 'raw', 'AirQualityData.csv')
    processed_path = os.path.join('data', 'processed', 'AirQualityData_cleaned.csv')
    
    if os.path.exists(raw_path):
        clean_air_quality_data(raw_path, processed_path)
    else:
        print(f"Error: {raw_path} not found.")
