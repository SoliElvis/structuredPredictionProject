FROM python:3.7-slim
# Set work directory
ADD . /project
WORKDIR ./project
Run pip install -r requirements.txt
Run ipython
