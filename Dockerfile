# Step 1: Python PyTorch Base Image
FROM python:3.11-slim

# Install system dependencies & Node.js for frontend build
RUN apt-get update && apt-get install -y curl && \
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Step 2: Install Python Backend Dependencies
COPY backend/requirements.txt ./backend/
RUN pip install --no-cache-dir -r ./backend/requirements.txt

# Step 3: Install Frontend Dependencies & Build Static Assets
COPY frontend/package*.json ./frontend/
RUN cd frontend && npm install

COPY . .
RUN cd frontend && npm run build

# Step 4: Configure Environment & Run Server
ENV PYTHONPATH=/app/backend
ENV PORT=10000

EXPOSE 10000

# Start Uvicorn bound to Render's assigned PORT
CMD ["sh", "-c", "python -m uvicorn app.main:app --host 0.0.0.0 --port $PORT"]
