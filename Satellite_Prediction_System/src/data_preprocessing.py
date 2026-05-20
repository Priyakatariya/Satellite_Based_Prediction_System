import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

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
    
    return combined_df

def clean_combined_data(combined_df=None):
    if combined_df is None:
        input_path = os.path.join('data', 'processed', 'combined_dataset.csv')
        if not os.path.exists(input_path):
            print("Combined dataset not found. Run merge_datasets() first.")
            return
        combined_df = pd.read_csv(input_path)
        
    print("Starting post-merge data cleaning...")
    
    # 1. Rename and drop duplicate columns
    # Pandas appends _x and _y to overlapping columns.
    # We keep Indian PM2.5 (pm2_5_x) as the main target and drop the UCI PM2.5 (pm2_5_y).
    if 'pm2_5_x' in combined_df.columns:
        combined_df = combined_df.rename(columns={'pm2_5_x': 'pm2_5'})
    if 'pm2_5_y' in combined_df.columns:
        combined_df = combined_df.drop(columns=['pm2_5_y'])
        
    # 2. Sort data by date and station
    if 'date' in combined_df.columns and 'stn_code' in combined_df.columns:
        combined_df['date'] = pd.to_datetime(combined_df['date'])
        combined_df = combined_df.sort_values(by=['stn_code', 'date']).reset_index(drop=True)
        
    # 3. Handle any NaN values created by the merge (Forward fill, then backward fill within each station)
    print("Handling missing values...")
    if 'stn_code' in combined_df.columns:
        # Avoid the DeprecationWarning regarding grouped fillna
        # Instead, we apply ffill and bfill directly on the dataframe groupby columns
        grouped = combined_df.groupby('stn_code')
        for col in combined_df.columns:
            if col not in ['stn_code']:
                combined_df[col] = grouped[col].transform(lambda x: x.ffill().bfill())
    else:
        combined_df = combined_df.ffill().bfill()
        
    # 4. Feature Scaling (Standardization)
    print("Scaling numerical features...")
    # We shouldn't scale the target variable ('pm2_5') or identifiers/categorical variables
    exclude_cols = ['date', 'stn_code', 'state', 'location', 'type', 'pm2_5', 'year', 'month', 'day']
    num_cols = [col for col in combined_df.select_dtypes(include=[np.number]).columns if col not in exclude_cols]
    
    scaler = StandardScaler()
    combined_df[num_cols] = scaler.fit_transform(combined_df[num_cols])
    
    output_path = os.path.join('data', 'processed', 'model_ready_data.csv')
    combined_df.to_csv(output_path, index=False)
    print(f"Model-ready dataset saved to {output_path}")
    print(f"Final shape: {combined_df.shape}")

if __name__ == "__main__":
    df = merge_datasets()
    clean_combined_data(df)
