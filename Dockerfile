FROM python:3.10-bookworm

WORKDIR /backend

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY ./backend .

CMD ["python3", "run.py"]