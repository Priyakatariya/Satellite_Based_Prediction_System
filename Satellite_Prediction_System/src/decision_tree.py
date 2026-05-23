import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# Load dataset
df = pd.read_csv("Satellite_Prediction_System/data/processed/model_ready_data.csv")

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

# Create model
model = DecisionTreeRegressor(
    max_depth=10,
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

print("===== Decision Tree Results =====")
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)

# Save model
joblib.dump(
    model,
    'Satellite_Prediction_System/models/decision_tree_model.pkl'
)

print("Decision Tree Model Saved")

# Visualization
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred)

plt.xlabel("Actual PM2.5")
plt.ylabel("Predicted PM2.5")
plt.title("Decision Tree: Actual vs Predicted")

plt.show()