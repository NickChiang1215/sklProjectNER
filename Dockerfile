FROM pytorch/pytorch:1.5-cuda10.1-cudnn7-devel

LABEL Name=skl_ner_service Version=0.0.1

ADD . /app
WORKDIR /app


#Using pip:
RUN python -m pip install --upgrade pip
RUN python -m pip install -r requirements.txt


# Define default command.
# CMD sh /app/mount/mount_in_docker.sh && /bin/bash
# CMD ["python","api.py", "-p"]
# RUN python api.py -p ${port}
CMD ["sh", "-c", "python ./api_service/app.py"]