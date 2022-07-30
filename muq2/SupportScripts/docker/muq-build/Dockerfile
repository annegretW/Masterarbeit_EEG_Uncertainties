FROM debian:latest
LABEL org.opencontainers.image.authors="M. Parno"

USER root

# Install most dependencies
RUN apt-get update && \
    apt-get install -y wget unzip cmake clang git g++ libgtest-dev libhdf5-dev libeigen3-dev libboost-all-dev libnlopt-dev libopenmpi-dev openmpi-bin libspdlog-dev doxygen python3-dev python3-pip pybind11-dev doxygen && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Make sure gtest libraries and header files are installed
RUN cd /usr/src/gtest && \
    cmake CMakeLists.txt && \
    make && \
    cp lib/*.a /usr/lib && cd ~

# Install nanoflann
RUN git clone --depth=1 https://github.com/jlblancoc/nanoflann.git && \
    cd nanoflann && mkdir build && cd build && \
    cmake .. && make install && \
    cd ../../ && rm -r nanoflann

# Install STANMATH
RUN wget https://github.com/stan-dev/math/archive/release/v2.18.0.zip && \
    unzip v2.18.0.zip && \
    cp -r math-release-v2.18.0/stan/ /usr/local/include/stan

# Install parcer
RUN git clone --depth=1 https://mparno@bitbucket.org/mituq/parcer.git && \
    cd parcer && mkdir build && cd build && \
    cmake .. && make -j2 && make install && \
    cd ../../ && rm -r parcer

# Install sundials 5.7 from source because packages are out of date
RUN wget https://github.com/LLNL/sundials/releases/download/v5.7.0/sundials-5.7.0.tar.gz && \
    tar -xzvf sundials-5.7.0.tar.gz && \
    cd sundials-5.7.0 && mkdir build && cd build && \
    cmake ../ && make -j2 && make install && \
    cd ../../ &&  rm -r sundials-5.7.0  &&  rm sundials-5.7.0.tar.gz

RUN useradd -ms /bin/bash  muq-user
WORKDIR /home/muq-user
USER muq-user

CMD ["bash"]
