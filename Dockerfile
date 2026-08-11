FROM python:3.13-slim

WORKDIR /app

# Instalar solo lo mínimo para torch CPU (sin CUDA, mucho más ligero)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# torch CPU-only (~200MB vs 2.5GB con CUDA)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Resto de dependencias desde PyPI normal
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código
COPY model.py server.py ./

HEALTHCHECK --interval=60s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:3000/health')" || exit 1

EXPOSE 3000

CMD ["python", "server.py"]
