# 🌍 Air Quality Prediction System using Machine Learning

## 📌 Project Overview
This project focuses on predicting **PM2.5 air pollution levels** using Machine Learning by combining:

- Ground monitoring station data
- Satellite imagery/data
- Weather parameters

The system aims to help in:
- Air quality forecasting
- Pollution monitoring
- Environmental analysis
- Smart city applications

The final product includes:
- Data preprocessing pipeline
- Machine learning models
- Interactive Streamlit dashboard
- Visualization and prediction system

---

# 🚀 Tech Stack

## 🖥️ Programming Language
- Python

## 📚 Libraries & Frameworks
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- Streamlit
- Joblib / Pickle

---

# 📂 Project Structure

```bash
Air-Quality-Prediction/
│
├── data/
│   ├── raw/
│   ├── cleaned/
│   └── processed/
│
├── notebooks/
│   ├── eda/
│   ├── preprocessing/
│   └── modeling/
│
├── models/
│   ├── best_model.pkl
│   ├── scaler.pkl
│   └── encoder.pkl
│
├── app/
│   ├── streamlit_app.py
│   └── prediction_pipeline.py
│
├── visualizations/
│
├── reports/
│   ├── documentation.pdf
│   └── presentation.pptx
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# 📊 Datasets Used

## 1️⃣ PM2.5 Ground Truth Data

### Possible Sources:
- CPCB India
- OpenAQ
- Kaggle Datasets

### Data Includes:
- PM2.5 concentration
- Monitoring station
- Date & time
- City/location

---

## 2️⃣ Satellite Data

### Possible Sources:
- NASA MODIS
- Sentinel Satellite Data

### Features:
- Aerosol Optical Depth (AOD)
- Surface reflectance
- Land information

---

## 3️⃣ Weather Data

### Possible Sources:
- ERA5 Weather Dataset
- OpenWeather API

### Features:
- Temperature
- Humidity
- Wind speed
- Pressure
- Rainfall

---

# ⚙️ Project Workflow

## ✅ Data Collection
- Collection of PM2.5 ground monitoring data
- Collection of satellite data
- Collection of weather data

---

## ✅ Data Understanding
- Missing value analysis
- Duplicate detection
- Statistical analysis
- Feature understanding

---

## ✅ Data Cleaning
- Null value handling
- Duplicate removal
- Datetime formatting
- Column standardization

---

## ✅ Dataset Integration
- Merging PM2.5 data with weather data
- Integrating satellite features
- Building final combined dataset

---

## ✅ Exploratory Data Analysis (EDA)

### Visualizations Performed:
- Distribution plots
- Histograms
- Boxplots
- Heatmaps
- Seasonal trend analysis
- Time-series analysis
- Correlation analysis

---

## ✅ Feature Engineering

### Features Created:
- Lag features
- Rolling averages
- Seasonal features
- Date-based features

---

# 🤖 Machine Learning Models Used

## 📌 Baseline Models
- Linear Regression
- Ridge Regression

## 📌 Tree-Based Models
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor

## 📌 Advanced Model
- XGBoost Regressor

---

# 📈 Model Evaluation Metrics

The following metrics are used for evaluation:

- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² Score
- Cross Validation Score

---

# 🔥 Hyperparameter Tuning

## Techniques Used
- GridSearchCV
- RandomizedSearchCV

## Purpose
- Improve model accuracy
- Reduce overfitting
- Optimize performance

---

# 🧠 Machine Learning Pipeline

```text
Data Collection
      ↓
Data Cleaning
      ↓
Feature Engineering
      ↓
EDA
      ↓
Model Training
      ↓
Hyperparameter Tuning
      ↓
Model Evaluation
      ↓
Deployment using Streamlit
```

---

# 🌐 Streamlit Dashboard Features

✅ PM2.5 prediction system  
✅ Interactive visualizations  
✅ Pollution trend analysis  
✅ Heatmaps  
✅ User input system  
✅ Real-time predictions  
✅ Clean and responsive UI  

---

# 📷 Visualizations Included

- Distribution plots
- Histograms
- Boxplots
- Heatmaps
- Seasonal trends
- Time-series graphs
- Pollution trend analysis

---

# ▶️ How to Run the Project

## 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/Air-Quality-Prediction.git
cd Air-Quality-Prediction
```

---

## 2️⃣ Install Requirements

```bash
pip install -r requirements.txt
```

---

## 3️⃣ Run Streamlit App

```bash
streamlit run app/streamlit_app.py
```

---

# 📌 Future Improvements

- Real-time API integration
- Deep Learning models (LSTM)
- Live satellite feed integration
- AQI forecasting
- Mobile application deployment
- Geo-spatial visualization
- Real-time pollution alerts

---

# 📄 Deliverables

✅ Final processed dataset  
✅ Trained ML model  
✅ Streamlit deployment  
✅ Visualization dashboard  
✅ Documentation  
✅ PPT Presentation  

---

# 📜 License

This project is created for educational and research purposes.

---

# ⭐ Acknowledgements

Special thanks to:
- CPCB India
- NASA MODIS
- Sentinel Data
- OpenAQ
- Scikit-learn Community
- Streamlit

---

# 💡 Conclusion

This project demonstrates how Machine Learning and environmental data can be combined to build an intelligent air quality prediction system capable of supporting smarter environmental monitoring and decision-making.

The system integrates PM2.5 monitoring data, satellite observations, and weather parameters to build accurate predictive models and provide meaningful visual insights through an interactive dashboard.

---