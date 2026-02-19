# ── Stage 1: Frontend build ──
FROM node:20-slim AS frontend-build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY index.html ./
COPY src/ ./src/
COPY public/ ./public/
COPY assets/ ./assets/
RUN npm run build

# ── Stage 2: Backend + serve frontend ──
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.11 + system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3-pip python3.11-dev \
    poppler-utils \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Install Node.js 20 (for Vite dev server)
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies in a virtual env (avoids system pip issues)
COPY backend/requirements.txt ./backend/requirements.txt
RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r backend/requirements.txt

# Install npm dependencies (cached layer)
COPY package.json package-lock.json ./
RUN npm ci

# Copy project files
COPY . .

# Copy built frontend from stage 1
COPY --from=frontend-build /app/dist ./dist

# HuggingFace cache directory (mounted as Docker volume)
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface

# Create directories for volumes
RUN mkdir -p /app/.cache/huggingface \
    && mkdir -p /app/backend/training_data \
    && mkdir -p /app/backend/adapters

EXPOSE 8000 5173

# Start both backend and frontend
CMD ["bash", "-c", "cd backend && python server.py & npm run dev -- --host 0.0.0.0 & wait"]
