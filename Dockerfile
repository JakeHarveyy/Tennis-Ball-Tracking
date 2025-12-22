# 1. Base Image
FROM python:3.10-slim

# 2. System Dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 3. Working Directory
WORKDIR /app

# 4. Install Python Dependencies
COPY requirements.txt .

# Install CPU-only PyTorch first (to keep image small)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install the rest of the requirements
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy Source Code
COPY . .

# 6. Expose Ports
EXPOSE 8000
EXPOSE 8501

# 7. Default Command
CMD ["bash", "-c", "uvicorn src.app.main:app --host 0.0.0.0 --port 8000 & streamlit run src/app/ui.py --server.port 8501 --server.address 0.0.0.0"]