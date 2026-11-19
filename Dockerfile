FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000 MAX_FPS=30 DISABLE_PROCESS_POOL=1
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --only-binary=:all: -r requirements.txt \
    && useradd --create-home --uid 10001 lidar
COPY prototype_app.py ./
COPY src/ src/
COPY static/ static/
USER lidar
EXPOSE 8000
CMD ["python", "prototype_app.py", "--host", "0.0.0.0"]
