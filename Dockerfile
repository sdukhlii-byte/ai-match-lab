FROM python:3.12-slim

# PYTHONUNBUFFERED — иначе логи Railway буферизуются и при падении контейнера
# последние строки (самые интересные) просто теряются.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libcairo2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Кэш CSV и папка результатов должны быть доступны на запись непривилегированному
# пользователю — контейнеру незачем работать от root.
RUN useradd --create-home --uid 10001 aml \
    && mkdir -p /app/out /app/.cache /app/state \
    && chown -R aml:aml /app
USER aml

CMD ["python", "generate.py", "--auto"]
