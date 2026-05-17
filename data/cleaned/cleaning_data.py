import pandas as pd
import numpy as np
import sklearn 

df = pd.read_csv(
    'data/raw/data.csv',
    encoding='latin1',
    low_memory=False
)

#handling missing values
df = df.dropna(subset=['pm2_5'])
df = df.drop(columns=['spm'])

df['so2'] = df['so2'].fillna(df['so2'].median())

df['no2'] = df['no2'].fillna(df['no2'].median())

df['rspm'] = df['rspm'].fillna(df['rspm'].median())
print(df.isnull().sum())
print(df.duplicated().sum())
#handling proper date format
df['date'] = pd.to_datetime(df['date'], errors='coerce')
print(df['date'].isnull().sum())
df['year'] = df['date'].dt.year

df['month'] = df['date'].dt.month

df['day'] = df['date'].dt.day
print(df[['date', 'year', 'month', 'day']].head())
#handling impossible values
print(df[['so2', 'no2', 'rspm', 'pm2_5']].min())
#handling outliersusing iqr method
print(df[['so2', 'no2', 'rspm', 'pm2_5']].describe())
Q1 = df['pm2_5'].quantile(0.25)

Q3 = df['pm2_5'].quantile(0.75)

IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR

upper = Q3 + 1.5 * IQR

print(lower, upper)
df = df[
    (df['pm2_5'] >= lower) &
    (df['pm2_5'] <= upper)
]
print(df.shape)
df = df.drop(columns=[
    'sampling_date',
    'agency',
    'location_monitoring_station'
])
#encoding the data
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

df['state'] = encoder.fit_transform(df['state'])

df['location'] = encoder.fit_transform(df['location'])

df['type'] = encoder.fit_transform(df['type'])
df.to_csv(
    'data/processed/cleaned_data.csv',
    index=False
)

print("Cleaned CSV saved successfully")
