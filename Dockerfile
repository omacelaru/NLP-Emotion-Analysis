# Folosim Python 3.11-slim ca imagine de bază
FROM python:3.11-slim

# Setăm directorul de lucru și variabilele de mediu
WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Instalăm dependențele și curățăm în același layer
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    pip install --no-cache-dir --upgrade pip && \
    rm -rf /var/lib/apt/lists/*

# Copiem doar requirements.txt mai întâi pentru a beneficia de cache
COPY requirements.txt .

# Instalăm dependențele Python
RUN pip install --no-cache-dir -r requirements.txt && \
    rm -rf /root/.cache/pip/*

# Copiem codul aplicației
COPY app/ ./app/

# Expunem portul
EXPOSE 8501

# Comanda de pornire
CMD ["streamlit", "run", "app/main.py"] 