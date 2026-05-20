# Production-Ready Credit Card Fraud Detection System

An end-to-end machine learning pipeline designed to detect fraudulent credit card transactions in real-time, handle extreme class imbalance, and maintain robust MLOps engineering practices.

## 🏗️ System Architecture & File Structure

fraud-detection/
├── .github/workflows/ci.yml  # Automated CI/CD Testing & Build Pipeline
├── data/                     # Data directory (Raw and Processed Splits)
├── notebooks/                # EDA & Statistical Analysis Notebooks
├── src/                      # Production Engine Source Code
│   ├── preprocessing.py      # Scaling & Stratified Splitting Engine
│   ├── training.py           # MLflow Experimentation & Serialization Suite
│   └── api.py                # Core FastAPI Inference REST Engine
├── models/                   # Serialized Best Production Model Artifacts
├── tests/                    # PyTest Unit Verification Scripts
├── Dockerfile                # Multi-stage Container Blueprints
└── requirements.txt          # Explicit Dependency Register

## 🚀 How to Run the Project Locally

### 1. Environment Setup & Preprocessing
```bash
# Install dependencies
pip install -r requirements.txt

# Run preprocessing to generate stratified 80/20 data splits
python src/preprocessing.py

2. Model Training & MLOps Tracking
Bash
# Execute the training sequence across 3 candidate architectures
python src/training.py

# Launch the MLflow UI Dashboard
python -m mlflow ui --backend-store-uri file:///C:/Users/chara/OneDrive/Desktop/fraud

3. Launching the REST API Engine
Bash
python -m uvicorn src.api:app --reload --host 127.0.0.1 --port 8000
Interactive documentation can be accessed locally at http://127.0.0.1:8000/docs.

🛠️ Automated CI/CD Pipeline
This project features an automated GitHub Actions pipeline that triggers on every push or pull_request to the main branch. It automatically installs dependencies, executes unit tests via pytest, and validates the multi-stage Docker container build.


---

### What to do next:
1. Create a `.gitignore` file in your project root folder and write `data/creditcard.csv` inside it. (This stops Git from uploading the massive 150MB raw file, while still allowing the smaller processed `.csv` files to go up).
2. Upload your entire project directory to your public **GitHub repository**.
3. Go to your repository's **Actions** tab on GitHub, watch your pipeline run, and take a screenshot of the successful green checkmarks! 

You now have your EDA notebook, preprocessing engine, trained production model, MLflow tracking history, operational FastAPI script, production Dockerfile, and CI/CD workflow. You are fully ready to submit your project URL!