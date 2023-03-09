FROM python:3.8.10

#RUN python3 -m venv /opt/venv

#RUN . /opt/venv/bin/activate && python3 -m pip install --upgrade pip

COPY ./analytics_lib /analytics_lib


#RUN pip install fastapi==0.92.0
#RUN pip install uvicorn==0.20.0

COPY requirements.txt requirements.txt
#RUN . /opt/venv/bin/activate && python3 -m pip install -r requirements.txt
#RUN . /opt/venv/bin/activate && python3 -m dostoevsky download fasttext-social-network-model
#RUN . /opt/venv/bin/activate && python3 -m spacy download ru_core_news_sm 
RUN python3 -m pip install -r requirements.txt
RUN python3 -m dostoevsky download fasttext-social-network-model
RUN python3 -m spacy download ru_core_news_sm 

#EXPOSE 8888

#WORKDIR /analytics_lib/nlp_texts
#ENTRYPOINT . /opt/venv/bin/activate && python3 -m jupyter lab --ip=0.0.0.0 --allow-root
WORKDIR ./
COPY analytics_lib/nlp_texts/main_api.py analytics_lib/nlp_texts/main_api.py
ENTRYPOINT  ["uvicorn", "analytics_lib.nlp_texts.main_api:app", "--host=0.0.0.0", "--port=8080"]