FROM python:3.7.0

RUN pip install --upgrade pip

COPY ./requirements.txt ./
RUN pip install -r requirements.txt
RUN rm ./requirements.txt

EXPOSE 5000

RUN mkdir -p /server
WORKDIR /server
# COPY ./wait-for-it.sh ./
# RUN chmod +x wait-for-it.sh

RUN mkdir /artifact/
RUN mkdir /mlruns

CMD mlflow server \
    --backend-store-uri ../mlruns \
    --default-artifact-root ../artifact \
    --host 0.0.0.0