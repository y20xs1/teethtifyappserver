FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY lib/ml /app/ml

CMD ["uvicorn", "ml.api:app", "--host", "0.0.0.0", "--port", "10000"] 