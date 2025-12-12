# Reproducible container for sports betting arbitrage analysis
FROM python:3.9-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System dependencies (for compiling scientific Python wheels if needed)
RUN apt-get update \ 
    && apt-get install -y --no-install-recommends build-essential \ 
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade pip \ 
    && pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY pyproject.toml setup.py README.md ./
COPY src ./src
COPY data ./data
COPY docs ./docs
COPY tests ./tests

# Install project in editable mode with dev extras for linting/type-checking
RUN pip install --no-cache-dir -e .[dev]

# Default working directory matches script expectations (run.py uses ../data paths)
WORKDIR /app/src

CMD ["python", "run.py"]
