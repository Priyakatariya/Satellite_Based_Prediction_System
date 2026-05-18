import pandas as pd
import numpy as np
import os

def merge_datasets():
    print("Loading datasets...")
    # 1. Load PM2.5 data (India data)
    pm25_path = os.path.join('data', 'processed', 'cleaned_data.csv')
    pm25_df = pd.read_csv(pm25_path)
    pm25_df['date'] = pd.to_datetime(pm25_df['date'])
    
    # 2. Load Weather and Satellite data (UCI Air Quality data)
    weather_path = os.path.join('data', 'processed', 'AirQualityData_cleaned.csv')
    weather_df = pd.read_csv(weather_path)
    weather_df['timestamp'] = pd.to_datetime(weather_df['timestamp'])
    
    # Extract date for merging
    weather_df['date'] = weather_df['timestamp'].dt.normalize()
    
    print("Aligning timelines for merging...")
    # To simulate the merging of these datasets for the hackathon, we align the years.
    # pm25_df has mostly 2015 data. weather_df has 2024 data.
    # We shift weather_df year to 2015 to create a rich combined dataset.
    def shift_year(d):
        if pd.isnull(d):
            return d
        try:
            return d.replace(year=2015)
        except ValueError:
            # Handle leap year issue (e.g. Feb 29, 2024 -> Feb 28, 2015)
            return d.replace(year=2015, day=28)
            
    weather_df['date'] = weather_df['date'].apply(shift_year)
    
    print("Aggregating weather & satellite data to daily averages...")
    # Aggregate weather data to daily level to match PM2.5 data grain
    num_cols = weather_df.select_dtypes(include=[np.number]).columns.tolist()
    # Drop hour/month/year columns if they exist to avoid confusion after aggregation
    cols_to_drop = [c for c in ['year', 'month', 'day', 'hour', 'dayofweek'] if c in num_cols]
    num_cols = [c for c in num_cols if c not in cols_to_drop]
    
    daily_weather = weather_df.groupby('date')[num_cols].mean().reset_index()
    
    print("Merging PM2.5 data with weather/satellite data...")
    # Merge datasets on date
    combined_df = pd.merge(pm25_df, daily_weather, on='date', how='inner')
    
    print(f"Combined dataset shape: {combined_df.shape}")
    
    # Save the combined dataset
    output_path = os.path.join('data', 'processed', 'combined_dataset.csv')
    combined_df.to_csv(output_path, index=False)
    print(f"Combined dataset saved successfully to {output_path}")

if __name__ == "__main__":
    merge_datasets()
