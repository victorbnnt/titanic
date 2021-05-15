FROM python:3.8.6-buster
COPY api /api
COPY titanic /titanic
COPY requirements.txt /requirements.txt
COPY models/GradientBoostingClassifier_v0.joblib models/GradientBoostingClassifier_v0.joblib
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
CMD uvicorn api.fast:app --host 0.0.0.0 --port $PORT
