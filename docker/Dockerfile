FROM mcr.microsoft.com/azureml/base-gpu:0.2.1

USER root
ENV HOME /root

RUN apt-get update && \
    apt-get install -y git wget unzip

RUN mkdir -p $HOME/.cache

RUN mkdir -p $HOME/protobuf

WORKDIR $HOME/protobuf

RUN wget -O protobuf.zip https://github.com/google/protobuf/releases/download/v3.0.0/protoc-3.0.0-linux-x86_64.zip && \
    unzip protobuf.zip

RUN mkdir -p $HOME/src

WORKDIR $HOME/src

RUN git clone https://github.com/tensorflow/models.git && \
    cd models && \
    git checkout r1.13.0

RUN cd models/research/ && \
    $HOME/protobuf/bin/protoc object_detection/protos/*.proto --python_out=.

ENV PYTHONPATH $PYTHONPATH:$HOME/src/models/research:$HOME/src/models/research/slim

WORKDIR /

COPY ./conda.yml .

RUN ldconfig /usr/local/cuda/lib64/stubs && \
    conda env create -p /azureml-envs/azureml -f conda.yml

RUN . /opt/miniconda/etc/profile.d/conda.sh && \
    conda activate /azureml-envs/azureml && \
    pip install Cython && \
    pip install pycocotools && \
    ldconfig

ENV PATH /azureml-envs/azureml/bin:$PATH
ENV LD_LIBRARY_PATH /azureml-envs/azureml/lib:$LD_LIBRARY_PATH

COPY ./lib/model_lib.py $HOME/src/models/research/object_detection

CMD ["bash"]