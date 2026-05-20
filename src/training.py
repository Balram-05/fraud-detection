import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import classification_report, precision_recall_curve, auc, confusion_matrix
from imblearn.over_sampling import SMOTE

def load_data(data_dir):
    """Loads preprocessed data splits."""
    X_train = pd.read_csv(os.path.join(data_dir, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(data_dir, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv")).values.ravel()
    y_test = pd.read_csv(os.path.join(data_dir, "y_test.csv")).values.ravel()
    return X_train, X_test, y_train, y_test

def evaluate_model(model, X_test, y_test):
    """Computes critical production metrics focusing heavily on Recall and PR-AUC."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate Precision-Recall Curve & Area Under Curve (PR-AUC)
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(recall, precision)
    
    # Generate standard classification reports
    report = classification_report(y_test, y_pred, output_dict=True)
    
    metrics = {
        "precision": report["1"]["precision"],
        "recall": report["1"]["recall"],
        "f1_score": report["1"]["f1-score"],
        "pr_auc": pr_auc
    }
    return metrics, y_pred

def run_experiment(run_name, model, X_tr, y_tr, X_te, y_te):
    """Wraps model execution, evaluation, and logging within an MLflow context."""
    with mlflow.start_run(run_name=run_name):
        print(f"\nTraining Experiment: {run_name}...")
        
        # Fit model
        model.fit(X_tr, y_tr)
        
        # Log basic parameters dynamically
        params = model.get_params()
        # Filter a few key parameters to avoid flooding the log terminal
        log_params = {k: params[k] for k in ["max_depth", "n_estimators", "C", "class_weight", "learning_rate"] if k in params}
        mlflow.log_params(log_params)
        
        # Evaluate & Log metrics
        metrics, _ = evaluate_model(model, X_te, y_te)
        mlflow.log_metrics(metrics)
        
        print(f"Metrics -> Recall: {metrics['recall']:.4f} | PR-AUC: {metrics['pr_auc']:.4f}")
        
        # Log the trained model as a usable artifact
        mlflow.sklearn.log_model(model, artifact_path="model")
        
        return metrics["pr_auc"], model

def main():
    # Setup pathing
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    data_dir = os.path.join(project_root, "data")
    models_dir = os.path.join(project_root, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    # Load dataset splits
    X_train, X_test, y_train, y_test = load_data(data_dir)
    
    # Configure MLflow local tracking repository
    mlflow.set_tracking_uri(f"file:///{os.path.join(project_root, 'mlruns')}")
    mlflow.set_experiment("Credit_Card_Fraud_Detection")
    
    best_pr_auc = 0.0
    best_model_obj = None

    # --- EXPERIMENT 1: Baseline Logistic Regression (No Sampling) ---
    model_lr = LogisticRegression(max_iter=1000, random_state=42)
    auc_lr, m_lr = run_experiment("Baseline_LogisticRegression", model_lr, X_train, y_train, X_test, y_test)
    if auc_lr > best_pr_auc: best_pr_auc, best_model_obj = auc_lr, m_lr

    # --- EXPERIMENT 2: Cost-Sensitive Random Forest (Adjusted Class Weights) ---
    model_rf = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight="balanced", random_state=42, n_jobs=-1)
    auc_rf, m_rf = run_experiment("CostSensitive_RandomForest", model_rf, X_train, y_train, X_test, y_test)
    if auc_rf > best_pr_auc: best_pr_auc, best_model_obj = auc_rf, m_rf

    # --- EXPERIMENT 3: XGBoost combined with SMOTE Oversampling ---
    print("\nApplying SMOTE balancing to training features...")
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    
    model_xgb = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1)
    auc_xgb, m_xgb = run_experiment("SMOTE_XGBoost", model_xgb, X_train_smote, y_train_smote, X_test, y_test)
    if auc_xgb > best_pr_auc: best_pr_auc, best_model_obj = auc_xgb, m_xgb

    # Serialize the absolute best-performing architecture for production API usage
    import pickle
    best_model_path = os.path.join(models_dir, "best_model.pkl")
    with open(best_model_path, "wb") as f:
        pickle.dump(best_model_obj, f)
    print(f"\n--- Strategy Execution Concluded! ---\nSaved top-tier architecture serialization to: {best_model_path}")

if __name__ == "__main__":
    main()