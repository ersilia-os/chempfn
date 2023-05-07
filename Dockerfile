FROM python:3.10.7-bullseye

WORKDIR .

COPY . .

# RUN apt-get install -y wget unzip
RUN python -m pip install --upgrade pip
RUN python -m pip install streamlit
# RUN wget https://github.com/ersilia-os/compound-embedding-lite/archive/refs/heads/main.zip -O compound-embedding-lite.zip
# RUN unzip ./compound-embedding-lite.zip
# RUN rm ./compound-embedding-lite.zip
RUN git clone https://github.com/ersilia-os/compound-embedding-lite/
RUN python -m pip install -e compound-embedding-lite/.
RUN python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
RUN git clone https://github.com/DhanshreeA/TabPFN.git
RUN python -m pip install -e TabPFN/.
RUN python -m pip install lolP==0.0.4

EXPOSE 8501
CMD ["streamlit", "run", "app/app.py"]