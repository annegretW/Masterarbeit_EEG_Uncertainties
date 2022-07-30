\page docker Using MUQ with Docker

# Docker
First, make sure you have docker installed.  Follow [these](https://docs.docker.com/get-started/) instructions if you don't.  We provide several MUQ docker images:

| Image Name | Description  |
|---|---|
| `mparno/muq`  | A simple image containing MUQ (c++ and python) installed in a debian environment with minimal dependencies.  |
| `mparno/muq-jupyter` | Same as the muq image, but with jupyter lab and several other python packages installed. |
| `mparno/muq-hippylib`  | An image building on the [FEniCS](https://fenicsproject.org/) `quay.io/fenicsproject/stable` image with additional installations of [hIPPylib](https://hippylib.github.io/), [hIPPylib2MUQ](https://github.com/hippylib/hippylib2muq), andd MUQ.  |
| `mparno/muq-build` | An image containing necessary dependencies for MUQ, but not MUQ itself.  This image is useful for continuous integration purposes and serves as the base for the `mparno/muq` and `mparno/muq-jupyter` images. |

The following command launches a container using the `muq` image and sharing the current directory with the `/home/muq-user` folder in the container:
```
docker pull muq
docker run -it -v ${PWD}:/home/muq-user muq bash
```

To launch a container using the `muq-jupyter` image, we should also share port 8888 so we can launch jupyter notebooks inside the container but view them in a host browser.   An example doing this is
```
docker run -it -p 8888:8888 -v ${PWD}:/home/muq-user muq-jupyter bash
```
