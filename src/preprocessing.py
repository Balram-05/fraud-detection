import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

def prepare_data():
    """
    Loads raw credit card transaction data, scales the 'Amount' feature,
    and performs a stratified 80/20 train-test split.
    """
    print("--- Starting Phase 2: Preprocessing & Feature Engineering ---")
    
    # Dynamically locate the project directory structure
    current_dir = os.path.dirname(os.path.abspath(__file__)) # src/
    project_root = os.path.dirname(current_dir)             # fraud_detection/
    
    data_path = os.path.join(project_root, "data", "creditcard.csv")
    output_dir = os.path.join(project_root, "data")
    
    # 1. Load Dataset
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Raw data file missing at: {data_path}\n"
                                f"Please make sure 'creditcard.csv' is placed inside your 'data/' folder.")
        
    df = pd.read_csv(data_path)
    print(f"Successfully loaded raw dataset. Shape: {df.shape}")
    
    # 2. Scale the 'Amount' feature using RobustScaler [cite: 22]
    print("Scaling 'Amount' feature via RobustScaler...")
    scaler = RobustScaler()
    df['Scaled_Amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
    
    # Drop original 'Amount' and 'Time'
    df = df.drop(['Amount', 'Time'], axis=1)
    
    # 3. Separate Features and Target
    X = df.drop(['Class'], axis=1)
    y = df['Class']
    
    # 4. Stratified 80/20 Train-Test Split [cite: 23]
    print("Executing Stratified 80/20 Train-Test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.20, 
        random_state=42, 
        stratify=y  # Ensures both sets contain identical fraud ratios
    )
    
    # 5. Save processed data segments as artifacts [cite: 24]
    print("Saving processed train/test splits to data directory...")
    X_train.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(output_dir, "y_test.csv"), index=False)
    
    print(f"Preprocessing Complete!\n"
          f"-> Train Set Size: {X_train.shape[0]} samples (Fraud: {y_train.sum()})\n"
          f"-> Test Set Size:  {X_test.shape[0]} samples (Fraud: {y_test.sum()})")

if __name__ == "__main__":
    prepare_data()