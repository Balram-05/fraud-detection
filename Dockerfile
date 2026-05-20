# --- Stage 1: Build & Dependency Resolution ---
FROM python:3.11-slim AS builder

WORKDIR /app

# Install system dependencies needed for compiling certain wheel packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and construct a standalone wheel directory
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt


# --- Stage 2: Minimal Final Production Runtime ---
FROM python:3.11-slim AS runner

WORKDIR /app

# Copy installed dependencies from the builder stage
COPY --from=builder /root/.local /root/.local
COPY --from=builder /app /app

# Ensure local user paths are mapped to Python's environment pathing
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1

# Copy only the code segments and model artifacts required for live inference
COPY src/ /app/src/
COPY models/ /app/models/

# Expose FastAPI's standard communication port
EXPOSE 8000

# Run Uvicorn directly to boot your API inside the container
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]