ARG CUDA="10.0"
ARG CUDNN="7"

FROM nvidia/cuda:${CUDA}-cudnn${CUDNN}-devel-ubuntu16.04

ENV DEBIAN_FRONTEND noninteractive

ADD requirements.txt /


RUN apt-get update; \
    apt-get install -y wget build-essential git sudo curl libgtk2.0-dev; \
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
# RUN conda install pytorch-nightly=1.0.0.dev20190328 cudatoolkit=${CUDA} \
#     torchvision==0.2.1 -c pytorch

# conda install pytorch-nightly=1.0.0.dev20190328 torchvision==0.2.1 -c pytorch

RUN conda install pytorch-nightly=1.0.0.dev20190328 cudatoolkit=${CUDA} \
    torchvision==0.2.1 -c pytorch

# RUN apt-get install -y apt-utils git curl ca-certificates bzip2 cmake tree htop bmon iotop g++ \
#  && apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev

# #installing torchvision from master
# RUN git clone https://github.com/pytorch/vision.git \
#  && cd vision \
#  && git checkout b45cdbfabba6d083077481d04b803a1627d9391f \
#  && python setup.py install

# Install cityscapes
# RUN git clone https://github.com/mcordts/cityscapesScripts.git \
# && cd cityscapesScripts/ \
# && python setup.py build_ext install

#install pycocotools
RUN git clone https://github.com/cocodataset/cocoapi.git \
 && cd cocoapi/PythonAPI \
 && python setup.py build_ext install

# install apex
# RUN git clone https://github.com/NVIDIA/apex.git \
#  && cd apex \
#  && python setup.py install --cuda_ext --cpp_ext

# install PyTorch Detection
# ARG FORCE_CUDA="1"
# ENV FORCE_CUDA=${FORCE_CUDA}

# RUN git clone https://github.com/facebookresearch/maskrcnn-benchmark.git \
#  && cd maskrcnn-benchmark \
#  && git checkout 192261db14d596da52905e91dc608bd4315552dc \
#  && python setup.py build develop

    # conda install -y pytorch=1.1 torchvision cudatoolkit=10.0 -c pytorch;\
    # pip install -r requirements.txt; \
    # rm requirements.txt;\
    # #Mask-RCNN
    # pip install ninja yacs cython matplotlib tqdm opencv-python;\
    # #   install pycocotools
    # mkdir root/maskrcnn-benchmark/;\
    # export INSTALL_DIR="/root/maskrcnn-benchmark/";\
    # cd $INSTALL_DIR;\
    # git clone https://github.com/cocodataset/cocoapi.git;\
    # cd cocoapi/PythonAPI;\
    # python setup.py build_ext install;\
    # #   install cityscapesScripts
    # cd $INSTALL_DIR;\
    # git clone https://github.com/mcordts/cityscapesScripts.git;\
    # cd cityscapesScripts/;\
    # python setup.py build_ext install;\
    # #   install apex
    # cd $INSTALL_DIR;\
    # git clone https://github.com/NVIDIA/apex.git;\
    # cd apex;\
    # python setup.py install --cuda_ext --cpp_ext;\
    # #   install PyTorch Detection
    # cd $INSTALL_DIR;\
    # git clone https://github.com/facebookresearch/maskrcnn-benchmark.git;\
    # cd maskrcnn-benchmark;\
    # #   symlinks
    # python setup.py build develop;\
    # unset INSTALL_DIR;

ADD .bash_mods /

RUN cat /.bash_mods >> /root/.bashrc; \
    echo 'conda activate adept-model\n' >> /root/.bashrc; \
    rm /.bash_mods

ENTRYPOINT /bin/bash

ENV CUDA_VISIBLE_DEVICES=0 
RUN python -c "import torch; print(torch.cuda.is_available())"

