import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from xgboost import XGBRegressor

# Load dataset
df = pd.read_csv(
    "data/processed/model_ready_data.csv"
)

# Drop unnecessary columns
df = df.drop(columns=['date'])

if 'stn_code' in df.columns:
    df = df.drop(columns=['stn_code'])

# Features and target
X = df.drop(columns=['pm2_5'])
y = df['pm2_5']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create XGBoost model
model = XGBRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("===== XGBoost Results =====")
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)

# Save model
joblib.dump(
    model,
    'models/xgboost_model.pkl'
)

print("XGBoost Model Saved")

# Feature Importance
importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
})

importance = importance.sort_values(
    by='Importance',
    ascending=False
)

print("\nTop Important Features:")
print(importance.head(10))

# Visualization
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred)

plt.xlabel("Actual PM2.5")
plt.ylabel("Predicted PM2.5")
plt.title("XGBoost: Actual vs Predicted")

plt.show()
