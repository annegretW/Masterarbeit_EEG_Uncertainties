\page muqinstall Installation

# Getting MUQ
MUQ currently supports Linux and OSX systems.  We do not currently support Windows, but let us know if that is important to you and we may consider it in the future.

On Linux or OSX, there are three primary ways of installing MUQ:
 1. Using the \c muq conda package from the conda-forge channel.  (See instructions below.)
 2. Using the \c muq docker image. (See \subpage docker )
 3. Installing MUQ from source. (See \subpage source_install )


## Conda
A conda package containing the c++ and python components of MUQ is available via conda-forge.   To install and use MUQ using conda, use
```
conda install -c conda-forge muq
```
If using MUQ with c++, you will also need to install CMake in order to link with MUQ:
```
conda install -c conda-forge muq cmake
```

You may run into conflicts when installing MUQ and Fenics with conda if they are installed one at a time.  To avoid this, we recommend installing MUQ and Fenics at the same time:
```
conda install -c conda-forge muq fenics
```


# Linking against MUQ in c++
MUQ leverages CMake to configure its build.  For a C++ project building on MUQ, CMake is therefore the most natural way to link to the MUQ libraries.   Here's a minimal of example of a <code>CMakeLists.txt</code> file for linking against MUQ.
@codeblock{cmake,CMakeLists.txt}
cmake_minimum_required(VERSION 3.10)

project(GaussianProcess_CO2)
set(CMAKE_CXX_STANDARD 17)

find_package(MUQ REQUIRED)
include_directories(${MUQ_INCLUDE_DIRS})

add_executable(my_muq_exe MyCode.cpp)
target_link_libraries(my_muq_exe ${MUQ_LIBRARIES} ${MUQ_LINK_LIBRARIES})
@endcodeblock

Compiling the <code>my_muq_exe</code> executable involves the typical CMake configure and build steps.  On the command line, this might be
@codeblock{bash,Bash}
mkdir build
cd build
cmake ..
make
@endcodeblock

> **_NOTE:_**  On newer Macs with the Apple M1 processor, you might have to set <code>-DCMAKE_OSX_ARCHITECTURES=x86_64</code> when running cmake.

> **_NOTE:_**  When using MUQ with MPI, you will need to tell CMake to use MPI compilers both when building MUQ and when compiling the examples.    For example, building the examples may require the following CMake command 
@codeblock{bash,Bash}
mkdir build
cd build
cmake -DCMAKE_CXX_COMPILER=mpic++ -DCMAKE_C_COMPILER=mpicc ..
make
@endcodeblock

> **_NOTE:_** If you install MUQ to a non-standard location, such as `~/Installations/MUQ_INSTALL`, you may need to tell CMake where to look for the MUQ configuration.   This can be accomplished by explicitly defining the `CMAKE_PREFIX_PATH` variable in your CMake command.  For example,
@codeblock{bash,Bash}
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=~/Installations/MUQ_INSTALL/lib/cmake/MUQ ..
make
@endcodeblock