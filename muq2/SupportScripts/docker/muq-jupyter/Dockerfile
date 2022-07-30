ARG tag
FROM mparno/muq:$tag
MAINTAINER M. Parno

# Install extra python packages for running jupyter notebooks that use MUQ
RUN pip3 install matplotlib \
                scipy \
                pandas \
                h5py \
                jupyterlab

ENV PATH "${PATH}:/home/muq-user/.local/bin"
CMD ["bash"]
