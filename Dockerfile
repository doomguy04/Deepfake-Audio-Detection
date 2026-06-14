# Use a lightweight official Python image
FROM python:3.11-slim

# Install system dependencies (libsndfile1 is required by librosa/soundfile)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements file first for caching
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and config
COPY .streamlit/ .streamlit/
COPY src/ src/
COPY saved_models/ saved_models/
COPY demo_data/ demo_data/
COPY app.py .

# Expose Streamlit default port
EXPOSE 8501

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Start the Streamlit application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
