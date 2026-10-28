FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

# ===== EDIT THESE TWO LINES PER LANGUAGE =====
ENV MODEL_ID=EYEDOL/Yoruba-ASRNEW
ENV LANGUAGE=yoruba
# =============================================

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/models

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the model into the image (public models only).
# If your model is PRIVATE, delete this line and set HF_TOKEN as an
# environment variable on the RunPod endpoint instead.
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('$MODEL_ID')"

COPY handler.py .

CMD ["python", "-u", "handler.py"]
