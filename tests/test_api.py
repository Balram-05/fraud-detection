import pytest
from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)

def test_health_endpoint():
    """Ensure the health check endpoint returns operational status."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_predict_endpoint_validation():
    """Ensure payload structural verification triggers validation failures correctly."""
    # Sending an incomplete array (only 3 elements instead of 29)
    response = client.post("/predict", json={"features": [1.0, 2.0, 3.0]})
    assert response.status_code == 400