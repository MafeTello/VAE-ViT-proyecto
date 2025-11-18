FROM python:3.11

WORKDIR /app

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código fuente
COPY src /app/src
COPY app.py /app/app.py
COPY models /app/models
COPY configs /app/configs
COPY test_images /app/test_images

EXPOSE 7860

CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]