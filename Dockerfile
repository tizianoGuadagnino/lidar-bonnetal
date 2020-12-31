# Use an official nvidia runtime as a parent image
FROM nvidia/cuda:10.1-devel-ubuntu18.04

CMD ["bash"]

# opengl things
ENV DEBIAN_FRONTEND "noninteractive"
# ENVIRONMENT STUFF FOR CUDA
RUN ls /usr/local/cuda/bin
ENV LD_LIBRARY_PATH /usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/usr/local/cuda/lib64
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/usr/local/lib
ENV PATH=/usr/local/cuda/bin:$PATH
ENV CUDA_ROOT /usr/local/cuda

# apt packages
RUN apt-get update && apt-get install -yqq  build-essential ninja-build \
  python3-dev python3-pip tig apt-utils curl git cmake unzip autoconf autogen \
  libtool mlocate zlib1g-dev python python3-numpy python3-wheel wget \
  software-properties-common openjdk-8-jdk libpng-dev  \
  libxft-dev vim meld sudo ffmpeg python3-pip libboost-all-dev \
  libyaml-cpp-dev git -y && updatedb

RUN mkdir -p /home/developer && \
    cp /etc/skel/.bashrc /home/developer/.bashrc && \
    echo "developer:x:${uid}:${gid}:Developer,,,:/home/developer:/bin/bash" >> /etc/passwd && \
    echo "developer:x:${uid}:" >> /etc/group && \
    echo "developer ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/developer && \
    chmod 0440 /etc/sudoers.d/developer 
# python packages
RUN pip3 install -U pip
RUN pip3 install numpy==1.14.0 \
                 torchvision==0.2.2.post3 \
                 matplotlib==2.2.3 \
                 tensorflow==1.13.1 \
                 scipy==0.19.1 \
                 torch==1.1.0 \
                 vispy==0.5.3 \
                 opencv_python==4.1.0.25 \
                 opencv_contrib_python==4.1.0.25 \
                 Pillow==6.1.0 \
                 PyYAML==5.1.1
# clean the cache
RUN apt update && \
  apt autoremove --purge -y && \
  apt clean -y
# Set the working directory to the api location
WORKDIR /home/developer/
# make user and home
USER developer
ENV HOME /home/developer
ADD . /home/developer/pcw-net

