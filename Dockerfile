FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and config
COPY . .

# OpenEnv wraps the environment in a FastAPI server on port 8000
EXPOSE 8000

# Start OpenEnv server based on the openenv.yaml specification
CMD ["openenv", "serve", "openenv.yaml", "--host", "0.0.0.0", "--port", "8000"]
