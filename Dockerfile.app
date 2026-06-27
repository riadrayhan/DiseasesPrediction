# ==============================================================================
# Microbiome Prediction — Web app container (Streamlit UI)
# ==============================================================================
# Deploys the interactive web frontend. For the CLI/library image see
# Dockerfile.predict; for the preprocessing pipeline see Dockerfile.
#
# Build:  docker build -f Dockerfile.app -t microbiome-predict-app .
# Run:    docker run -p 8501:8501 microbiome-predict-app
# Then open http://localhost:8501
# ==============================================================================
FROM python:3.11-slim

LABEL maintainer="Riad Rayhan"
LABEL description="Microbiome disease-prediction Streamlit web app"

WORKDIR /app

# libgomp1 is needed by scikit-learn / xgboost (OpenMP).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501
ENV STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_BROWSER_GATHERUSAGESTATS=false

HEALTHCHECK CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8501/_stcore/health').read()==b'ok' else 1)" || exit 1

CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
