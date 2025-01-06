# Используем официальный образ Python 3.9
FROM python:3.9-slim
RUN pip install --upgrade poetry
COPY . .
WORKDIR ./
RUN poetry install --no-root
CMD ["sh", "-c", "poetry run python train.py && poetry run python infer.py"]
