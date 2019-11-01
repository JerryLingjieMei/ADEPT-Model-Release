ARG CUDA="10.0"
ARG CUDNN="7"

FROM nvidia/cuda:${CUDA}-cudnn${CUDNN}-devel-ubuntu16.04

ENV DEBIAN_FRONTEND noninteractive

ADD requirements.txt /


RUN apt-get update; \
    apt-get install -y wget build-essential git sudo curl libgtk2.0-dev ffmpeg; \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh; \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /root/miniconda;

ENV PATH=/root/miniconda/bin:$PATH

RUN conda init bash

RUN conda create -y -n adept-model python=3.6.9;

ENV CONDA_DEFAULT_ENV=adept-model
ENV CONDA_PREFIX=/root/miniconda/envs/$CONDA_DEFAULT_ENV
ENV PATH=$CONDA_PREFIX/bin:$PATH
ENV CONDA_AUTO_UPDATE_CONDA=false

RUN conda install -y pip cython numpy requests ipython

RUN pip install -r requirements.txt;\
    rm requirements.txt

RUN pip install requests ninja yacs cython matplotlib opencv-python tqdm

#install pytorch
ARG CUDA

RUN conda install pytorch-nightly=1.0.0.dev20190328 cudatoolkit=${CUDA} \
    torchvision==0.2.1 -c pytorch

RUN conda install -y jupyter



#install pycocotools
RUN git clone https://github.com/cocodataset/cocoapi.git \
 && cd cocoapi/PythonAPI \
 && python setup.py build_ext install


ADD .bash_mods /

RUN cat /.bash_mods >> /root/.bashrc; \
    echo 'conda activate adept-model\n' >> /root/.bashrc; \
    rm /.bash_mods

ENTRYPOINT /bin/bash

