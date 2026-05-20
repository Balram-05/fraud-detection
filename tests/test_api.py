import pytest
import os
from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)

def test_debug_environment_paths():
    """Diagnostic step to print out the exact cloud workspace structure."""
    print("\n========= CI/CD PATH DEBUGGING =========")
    print(f"Current Working Directory (CWD): {os.getcwd()}")
    print(f"Files in CWD: {os.listdir(os.getcwd())}")
    
    # Check if models folder exists and list its content
    if os.path.exists("models"):
        print(f"Files inside 'models/': {os.listdir('models')}")
    else:
        print("WARNING: 'models/' directory does not exist in root!")
        
    print("========================================\n")
    assert True

def test_health_endpoint():
    """Ensure the health check endpoint returns operational status."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_predict_endpoint_validation():
    """Ensure payload structural verification triggers validation failures correctly."""
    response = client.post("/predict", json={"features": [1.0, 2.0, 3.0]})
    assert response.status_code == 400