import os
import pickle
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_VIS = True
except ImportError:
    HAS_VIS = False


def load_data(file_path: str) -> pd.DataFrame:
    print("=" * 80)
    print(f"[STEP 1] LOADING DATASET FROM: {file_path}")
    print("=" * 80)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Error: Dataset not found at: {file_path}")
    df = pd.read_csv(file_path)
    print(f"-> Dataset loaded successfully! Shape: {df.shape[0]} rows, {df.shape[1]} columns.\n")
    return df


def preprocess_data(df: pd.DataFrame, target_col: str = 'pm2_5'):
    print("=" * 80)
    print("[STEP 2] PREPROCESSING & SCALING DATA")
    print("=" * 80)
    
    processed_df = df.copy()
    columns_to_drop = ['date', 'stn_code']
    for col in columns_to_drop:
        if col in processed_df.columns:
            processed_df = processed_df.drop(columns=[col])
            
    X = processed_df.drop(columns=[target_col])
    y = processed_df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("-> Fitting and applying StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("-> Feature scaling completed successfully!\n")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def train_decision_tree(X_train: np.ndarray, y_train: pd.Series) -> DecisionTreeRegressor:
    print("=" * 80)
    print("[STEP 3A] TRAINING DECISION TREE")
    print("=" * 80)
    start_time = time.time()
    model = DecisionTreeRegressor(max_depth=10, random_state=42).fit(X_train, y_train)
    duration_ms = (time.time() - start_time) * 1000
    print(f"-> Decision Tree trained successfully in {duration_ms:.2f} ms!\n")
    return model


def train_random_forest(X_train: np.ndarray, y_train: pd.Series) -> RandomForestRegressor:
    print("=" * 80)
    print("[STEP 3B] TRAINING RANDOM FOREST")
    print("=" * 80)
    start_time = time.time()
    model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1).fit(X_train, y_train)
    duration_ms = (time.time() - start_time) * 1000
    print(f"-> Random Forest trained successfully in {duration_ms:.2f} ms!\n")
    return model


def train_gradient_boosting(X_train: np.ndarray, y_train: pd.Series) -> GradientBoostingRegressor:
    print("=" * 80)
    print("[STEP 3C] TRAINING GRADIENT BOOSTING")
    print("=" * 80)
    start_time = time.time()
    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42).fit(X_train, y_train)
    duration_ms = (time.time() - start_time) * 1000
    print(f"-> Gradient Boosting trained successfully in {duration_ms:.2f} ms!\n")
    return model


def evaluate_and_compare(models_dict: dict, X_test: np.ndarray, y_test: pd.Series) -> dict:
    print("=" * 80)
    print("[STEP 4] COMPARING TREE MODELS")
    print("=" * 80)
    
    metrics = {}
    print(f"   {'Model Name':<25} | {'MAE':<10} | {'MSE':<12} | {'RMSE':<10} | {'R2 Score':<10}")
    print("   " + "-" * 75)
    for name, model in models_dict.items():
        y_pred = model.predict(X_test)
        metrics[name] = {
            'mae': mean_absolute_error(y_test, y_pred),
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': r2_score(y_test, y_pred),
            'predictions': y_pred
        }
        print(f"   {name:<25} | {metrics[name]['mae']:<10.4f} | {metrics[name]['mse']:<12.4f} | {metrics[name]['rmse']:<10.4f} | {metrics[name]['r2']:<10.4f}")
    print("\n")
    return metrics


def save_artifacts(models_dict: dict, scaler: StandardScaler, output_dir: str = 'models'):
    print("=" * 80)
    print("[STEP 5] SAVING PIPELINE ARTIFACTS")
    print("=" * 80)
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
        
    for name, model in models_dict.items():
        safe_name = name.lower().replace(' ', '_')
        with open(os.path.join(output_dir, f'{safe_name}_model.pkl'), 'wb') as f:
            pickle.dump(model, f)
    print("-> All tree model artifacts saved!\n")


def generate_comparison_plots(y_test: pd.Series, metrics: dict, output_dir: str = 'reports/figures'):
    if not HAS_VIS: return
    print("=" * 80)
    print("[STEP 6] GENERATING DIAGNOSTIC COMPARISON PLOTS")
    print("=" * 80)
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="darkgrid")
    
    # R2 comparison
    plt.figure(figsize=(10, 5))
    names = list(metrics.keys())
    r2_scores = [metrics[n]['r2'] for n in names]
    sns.barplot(x=r2_scores, y=names, palette='viridis')
    plt.xlabel('R² Score', fontweight='bold')
    plt.title('Tree Models Comparison - R² Score', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'tree_models_r2_comparison.png'), dpi=300)
    plt.close()
    
    # Side-by-side scatter
    fig, axes = plt.subplots(1, 3, figsize=(24, 7), sharey=True)
    
    dt_preds = metrics['Decision Tree']['predictions']
    rf_preds = metrics['Random Forest']['predictions']
    gb_preds = metrics['Gradient Boosting']['predictions']
    
    # Decision Tree
    axes[0].scatter(y_test, dt_preds, alpha=0.5, color='#ef4444', edgecolors='k', s=40, label='DT Predictions')
    lims = [min(y_test.min(), dt_preds.min()), max(y_test.max(), dt_preds.max())]
    axes[0].plot(lims, lims, 'k--', alpha=0.8, linewidth=2, label='Perfect Fit')
    axes[0].set_xlabel('Actual PM2.5 (ug/m3)', fontweight='bold')
    axes[0].set_ylabel('Predicted PM2.5 (ug/m3)', fontweight='bold')
    axes[0].set_title('Decision Tree: Actual vs Predicted', fontweight='bold')
    axes[0].legend()
    
    # Random Forest
    axes[1].scatter(y_test, rf_preds, alpha=0.5, color='#10b981', edgecolors='k', s=40, label='RF Predictions')
    lims = [min(y_test.min(), rf_preds.min()), max(y_test.max(), rf_preds.max())]
    axes[1].plot(lims, lims, 'k--', alpha=0.8, linewidth=2, label='Perfect Fit')
    axes[1].set_xlabel('Actual PM2.5 (ug/m3)', fontweight='bold')
    axes[1].set_title('Random Forest: Actual vs Predicted', fontweight='bold')
    axes[1].legend()

    # Gradient Boosting
    axes[2].scatter(y_test, gb_preds, alpha=0.5, color='#f59e0b', edgecolors='k', s=40, label='GB Predictions')
    lims = [min(y_test.min(), gb_preds.min()), max(y_test.max(), gb_preds.max())]
    axes[2].plot(lims, lims, 'k--', alpha=0.8, linewidth=2, label='Perfect Fit')
    axes[2].set_xlabel('Actual PM2.5 (ug/m3)', fontweight='bold')
    axes[2].set_title('Gradient Boosting: Actual vs Predicted', fontweight='bold')
    axes[2].legend()
    
    plt.suptitle('Tree Models Comparison: Actual vs. Predicted PM2.5 Values', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    scatter_path = os.path.join(output_dir, 'tree_actual_vs_predicted.png')
    plt.savefig(scatter_path, dpi=300)
    plt.close()

    print("-> Plots saved successfully!\n")


def main():
    dataset_path = 'data/processed/model_ready_data.csv'
    try:
        df = load_data(dataset_path)
        X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
        
        models = {
            'Decision Tree': train_decision_tree(X_train, y_train),
            'Random Forest': train_random_forest(X_train, y_train),
            'Gradient Boosting': train_gradient_boosting(X_train, y_train)
        }
        
        metrics = evaluate_and_compare(models, X_test, y_test)
        save_artifacts(models, scaler)
        generate_comparison_plots(y_test, metrics)
        
        print("Success: Tree Models Pipeline completed successfully!\n" + "=" * 80)
    except Exception as e:
        print(f"Error: Pipeline failed with error: {str(e)}")

if __name__ == '__main__':
    main()
