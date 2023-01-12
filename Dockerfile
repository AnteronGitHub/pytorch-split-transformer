FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

RUN pip install torchdata==0.3.0

WORKDIR /app
COPY . .
CMD ["python3", "main.py"]
