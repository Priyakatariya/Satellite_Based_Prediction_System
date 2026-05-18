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
def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Summer'
    elif month in [6, 7, 8]:
        return 'Monsoon'
    else:
        return 'Post-Monsoon'
    
df['season'] = df['month'].apply(get_season)
from sklearn.preprocessing import LabelEncoder

season_encoder = LabelEncoder()

df['season'] = season_encoder.fit_transform(df['season'])
df['so2_no2_ratio'] = df['so2'] / (df['no2'] + 1)
df['total_pollution'] = (
    df['so2'] +
    df['no2'] +
    df['rspm']
)
print(df.head())
# Exploratory Data Analysis
import matplotlib.pyplot as plt
import seaborn as sns
#PM2.5 distribution plot##############
plt.figure(figsize=(10,6))
sns.histplot(df['pm2_5'],bins=20,kde=True)
plt.title('PM2.5 Distribution')
plt.xlabel('PM2.5')
plt.ylabel('Frequency')
plt.show()

#NO2 distribution plot ##############
plt.figure(figsize=(8,5))
sns.histplot(df['no2'], kde=True)
plt.title("NO2 Distribution")
plt.show()

#SO2 distribution plot ##############
plt.figure(figsize=(8,5))
sns.histplot(df['so2'],kde=True)
plt.title("So2 Distribution")
plt.show()


#HISTOGRAM
df[['so2','no2','rspm','pm2_5']].hist(
    figsize=(12,8),
    bins=20
)
plt.suptitle("Histograms of Pollutants")
plt.show()


#BOXPLOT( FOR OUTLIERS):
plt.figure(figsize=(10,6))
sns.boxplot(data=df[['so2','no2','rspm','pm2_5']])
plt.title("Boxplot of Pollutants")
plt.show()

#CORRELATION HEATMAP
plt.figure(figsize=(10,6))
sns.heatmap(df[['so2','no2','rspm','pm2_5']].corr(),annot=True,cmap='coolwarm')
plt.title("Correlation Heatmap of Pollutants")
plt.show()

#PM2.5 over time
plt.figure(figsize=(14,6))
plt.plot(df['date'], df['pm2_5'])
plt.title("PM2.5 Over Time")
plt.xlabel("Date")
plt.ylabel("PM2.5")
plt.show()