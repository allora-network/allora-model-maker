FROM python:3.11-slim AS project_env

# Install system dependencies including build essentials
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    gcc \
    g++

# Set the working directory in the container
WORKDIR /app

# Install Python build dependencies first
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip setuptools wheel \
    && pip install Cython==3.0.11 numpy==1.24.3 \
    && pip install -r requirements.txt

FROM project_env

COPY . /app/

# Set the entrypoint command
CMD ["gunicorn", "--conf", "/app/gunicorn_conf.py", "main:app"]
