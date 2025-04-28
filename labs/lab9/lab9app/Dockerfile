FROM python:3.13-slim

RUN pip install -U pip 

WORKDIR /app

COPY [ "requirements.txt", "./" ]

RUN pip install -r requirements.txt

COPY [ "reddit_model_pipeline.joblib", "reddit_app.py", "./"]

EXPOSE 8000

ENTRYPOINT [ "uvicorn", "reddit_app:app", "--host", "0.0.0.0"]