FROM python:3.11-slim

WORKDIR /app
COPY docker/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src /app
COPY models /app/models
COPY test_images /app/test_images

EXPOSE 7860
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]