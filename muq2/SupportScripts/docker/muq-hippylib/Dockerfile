FROM quay.io/fenicsproject/stable
LABEL org.opencontainers.image.authors="M. Parno"

# Run with "docker build --build-arg tag=v0.3.1 -t mparno/muq-hippylib:v0.3.1 ."
ARG tag

USER root

# Install most dependencies
RUN apt-get update && \
    apt-get install -y wget unzip cmake clang git g++ libgtest-dev libhdf5-dev libeigen3-dev libboost-all-dev libnlopt-dev libopenmpi-dev openmpi-bin libspdlog-dev doxygen python3-dev python3-pip pybind11-dev doxygen && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

USER fenics

# Install MUQ
RUN cd /home/fenics && \
    mkdir Installations; mkdir Installations/MUQ_INSTALL && \
    git clone --depth 1 --branch $tag https://bitbucket.org/mituq/muq2.git && \
    cd muq2/; mkdir build; cd build;  \
    cmake -DCMAKE_INSTALL_PREFIX=/home/fenics/Installations/MUQ_INSTALL -DMUQ_USE_PYTHON=ON -DMUQ_HDF5_DIR=/usr/local/lib/python3.5/dist-packages/h5py/.libs/ ../ && \
    make install && \
    cd /home/fenics && \
    rm -r muq2

# Install hIPPylib and hippylib2muq
RUN cd /home/fenics/Installations && \
    git clone https://github.com/hippylib/hippylib.git && \
    chmod -R o+rx hippylib && \
    git clone https://github.com/hippylib/hippylib2muq.git && \
    chmod -R o+rx hippylib2muq

# Install extra python packages for running jupyter notebooks that use MUQ
RUN pip3 install --user matplotlib  scipy  pandas jupyterlab

# Configure jupyter notebooks
ENV LD_LIBRARY_PATH /home/fenics/Installations/MUQ_INSTALL/lib:/home/fenics/Installations/MUQ_INSTALL/muq_external/lib
ENV PYTHONPATH /home/fenics/Installations/MUQ_INSTALL/python:/home/fenics/Installations/hippylib:/home/fenics/Installations/hippylib2muq

USER root

# Install h5py with the MUQ-installed HDF5 library
RUN cd /home/fenics/Installations && \
    git clone --depth=1 https://github.com/h5py/h5py.git && \
    cd h5py && \
    HDF5_DIR=/home/fenics/Installations/MUQ_INSTALL/muq_external pip install --no-binary=h5py h5py 

RUN mkdir -p /home/fenics/.jupyter


WORKDIR /home/fenics/
CMD ["bash"]
