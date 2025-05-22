FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime
RUN apt-get update && \
    apt-get install -y libgl1 libglib2.0-0 git && \
    rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .
COPY ava+logos-l14-linearMSE.pth .
EXPOSE 8080
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "2"]
