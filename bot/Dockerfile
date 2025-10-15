FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt


COPY . .
# Copy .env inside container


#CMD ["python", "bot.py"]
EXPOSE 8501
ENV PYTHONUNBUFFERED=1
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]