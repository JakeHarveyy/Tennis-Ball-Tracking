# 1. Base Image
FROM python:3.10-slim

# 2. System Dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 3. Setup User (Hugging Face Spaces requirement for security)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# 4. Working Directory
WORKDIR /app

# 5. Install Python Dependencies
# Copy requirements first to leverage cache
COPY --chown=user ./requirements.txt requirements.txt

# Install CPU-only PyTorch first (to keep image small)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install the rest of the requirements
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy Source Code
COPY --chown=user . .

# 7. Create necessary directories with correct permissions
RUN mkdir -p uploads data/outputs && chmod 777 uploads data/outputs

# 8. Expose Ports
# Hugging Face Spaces expects 7860. Heroku ignores EXPOSE but provides $PORT.
EXPOSE 7860

# 9. Default Command
# We run FastAPI on localhost:8000 (internal) and Streamlit on $PORT (public)
# If $PORT is not set (e.g. local or HF), default to 7860
# We disable CORS and XSRF protection because HF Spaces runs behind a proxy which causes 403 errors
CMD ["bash", "-c", "uvicorn src.app.main:app --host 0.0.0.0 --port 8000 & streamlit run src/app/ui.py --server.port ${PORT:-7860} --server.address 0.0.0.0 --server.enableCORS false --server.enableXsrfProtection false --server.maxUploadSize 500"]