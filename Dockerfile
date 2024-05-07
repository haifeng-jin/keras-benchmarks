FROM nvidia/cuda:12.0.0-devel-ubuntu20.04

# Set environment variables
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y \
        wget \
        build-essential \
        libssl-dev \
        zlib1g-dev \
        libncurses5-dev \
        libncursesw5-dev \
        libreadline-dev \
        libsqlite3-dev \
        libgdbm-dev \
        libdb5.3-dev \
        libbz2-dev \
        libexpat1-dev \
        liblzma-dev \
        tk-dev \
        libffi-dev \
        uuid-dev \
        libffi-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Download Python 3.10 source code
WORKDIR /tmp
RUN wget https://www.python.org/ftp/python/3.10.2/Python-3.10.2.tgz && \
    tar xzf Python-3.10.2.tgz

# Build and install Python 3.10
WORKDIR /tmp/Python-3.10.2
RUN ./configure --enable-optimizations && \
    make -j $(nproc) && \
    make altinstall

# Clean up
WORKDIR /
RUN rm -rf /tmp/Python-3.10.2*

# Install pip for Python 3.10
RUN wget https://bootstrap.pypa.io/get-pip.py | python3.10

# Set Python 3.10 as the default Python version
RUN ln -s /usr/local/bin/python3.10 /usr/local/bin/python3

# Install virtualenv
RUN pip3.10 install virtualenv

RUN pip3.10 install nvidia-pyindex
RUN pip3.10 install nvidia-cudnn

# Install vi 
RUN apt-get update && \
    apt-get install -y busybox && \
    ln -s /bin/busybox /bin/vi

ENTRYPOINT ["tail", "-f", "/dev/null"]
