FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

WORKDIR /app

# RUN apt-get update && apt-get install -y awscli

COPY [ "Pipfile", "Pipfile.lock", "./" ]

RUN pip install pipenv 

# RUN pip install s3fs ffspec
RUN pipenv install --system --deploy

COPY  "starter.py" "./"

ENTRYPOINT [ "python", "starter.py" ]