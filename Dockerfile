FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         cmake \
         git \
         curl \
         vim \
         ca-certificates \
         libjpeg-dev \
         libpng-dev &&\
     rm -rf /var/lib/apt/lists/*


RUN curl -o ~/miniconda.sh -O  https://repo.anaconda.com/archive/Anaconda3-5.3.1-Linux-x86_64.sh  && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh && \
     /opt/conda/bin/conda install -c cpbotha magma-cuda10 && \
     /opt/conda/bin/conda clean -ya
ENV PATH /opt/conda/bin:$PATH
RUN conda install pytorch torchvision cuda100 -c pytorch
RUN conda install numba
RUN pip install qpth
RUN pip install cvxpy
RUN pip install block
RUN apt-get update
RUN apt-get install -y --no-install-recommends sshpass
RUN pip install python-louvain
RUN pip install networkx --upgrade
RUN pip install autograd
RUN pip install python-igraph
RUN pip install gensim
RUN pip install textable
WORKDIR /
