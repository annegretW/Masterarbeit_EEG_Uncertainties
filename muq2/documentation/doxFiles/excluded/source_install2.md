```

## Configuring the Build
MUQ is written in c++ with a python wrapper provided through <a href=https://pybind11.readthedocs.io>pybind11</a>.   CMake is used to configure the MUQ build and generate makefiles.   Like most projects based on CMake, a basic configuration of MUQ can be accomplished by running CMake and only specifying the installation path.  For example, from the `muq2/build` directory, run
```
cmake -DCMAKE_INSTALL_PREFIX=/my/install/path ..
```

During this call, CMake will search for all necessary dependencies, test those dependencies, summarize the configuration in a file called `summary.log`, and create a makefile capable of compiling the MUQ libraries.

Additional configuration options can also be added to the CMake command to control the build process; the most important of these are described in the subsections below.

Note that after installation, you will need to update your environment variables to include the MUQ libraries (and optionally python bindings).    

On OSX:
```bash
export PYTHONPATH=$PYTHONPATH:/my/install/path/python:/my/install/path/lib
export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:/my/install/path/lib:/my/install/path/muq_external/lib
```

On Linux:
```bash
export PYTHONPATH=$PYTHONPATH:/my/install/path/python:/my/install/path/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/my/install/path/lib:/my/install/path/muq_external/lib
```

### Handling Dependencies
Various parts of MUQ rely on several external dependencies.  MUQ will download and build any required dependencies that are not found on your system and needed to compile the requested components of MUQ (see compile group discussion below).   The table below lists MUQ's dependencies, the version that will be compiled internally if the package is not found, as well as the name used to refer to the package in MUQ's CMake configuration scripts.

| Dependency | Internal Version | MUQ CMake Name |
| ---------- | -------------------- | -------------- |
| <a href=http://www.boost.org/>Boost</a> | 1.74 | `BOOST` |
| <a href=http://eigen.tuxfamily.org/>Eigen</a> | 3.3.7 | `EIGEN3` |
| <a href=http://www.hdfgroup.org/HDF5/>HDF5</a> | 1.8.19 | `HDF5` |
| <a href=http://mc-stan.org/>Stan Math</a> | 2.18.0 | `STANMATH` |
| <a href=https://computation.llnl.gov/projects/sundials>Sundials</a> | 5.4.0 | `SUNDIALS` |
| <a href=https://github.com/jlblancoc/nanoflann>nanoflann</a> | master | `NANOFLANN` |
| (Optional) <a href=https://github.com/google/googletest>GoogleTest</a> | N/A | `GTEST` |
| (Optional) Python | N/A | `PYTHON` |
| (Optional) MPI | N/A | `MPI` |

MUQ's CMake scripts follow a "find-check-build" process for each dependency.  First, CMake will search for existing installations of the required packages using the `find_package` function.   Simple compilation checks will then be performed for any found packages.  Finally, if a dependency was not found or did not pass the compilation checks, the CMake scripts will set up an external project to build the dependency from source.   The options listed below can optionally be passed to CMake to control this "find-check-build" process.  The `<name>` placeholder refers to "MUQ CMake Name" values in the table above.

| CMAKE Variable | Values | Description |
| -------------- | ------ | ----------- |
| `MUQ_<name>_DIR` | Path to folder containing `include` and `lib`| Used to specify the location of existing dependency installations on your machine.  This variable should be a path pointing the folder containing the `include` and `lib` folders for a specific dependency (e.g., `/usr/local` not `/usr/local/include`). |
| `MUQ_USE_<name>` | `ON` or `OFF` | Used to specify whether an option dependencies should be used or not. |
| `<name>_EXTERNAL_SOURCE` | Path or URL to tarball or repository. | If CMake cannot find an external dependency, the dependency will be built from source and installed in a muq-specific installation directory.  This CMake variable gives users the ability to specify the source code that should be used.  If not provided, default URL's for the versions listed above are used. |
| `MUQ_FORCE_INTERNAL_<name>` | `ON` or `OFF` | Forces CMake to build external dependencies from source whether or not existing versions of the packages are found. |

Some other useful general CMake options.
| CMAKE Variable | Values | Description |
| -------------- | ------ | ----------- |
| `CMAKE_INSTALL_PREFIX` | Any valid path | Used to specify the folder where MUQ will be installed.  `/usr/local` by default. |
| `CMAKE_CXX_COMPILER` | Name of compiler executable | Used to specify the compiler used to build MUQ.   Examples include `clang++`, `g++`, or `mpic++`.  Full paths to the compilers can also be included. |
| `CMAKE_C_COMPILER` | Name of compiler executable | Same as `CMAKE_CXX_COMPILER` but used for the c compiler.   The c compiler is used to compile some of MUQ's dependencies like `NLOPT` and `SUNDIALS`.   Examples include `clang`, `gcc`, or `mpicc`.|
| `CMAKE_BUILD_TYPE` | `RELEASE` or `DEBUG` | Defaults to `RELEASE` in MUQ.  Used to specify whether MUQ should be built with compiler optimizations (release mode) or debug symbols (debug mode). |

### Example CMake configurations:
All of the examples below assume that CMake is being run in a bash terminal from the `muq2/build` directory.  </br>

__Example 1:__ Specify the install directory, compile python bindings, and enable testing with GTest:
```
cmake                                                  \
  -DCMAKE_INSTALL_PREFIX=~/Installations/MUQ_INSTALL   \
  -DMUQ_USE_PYTHON=ON                                  \
  -DMUQ_USE_GTEST=ON                                   \
  -DMUQ_GTEST_DIR=~/Installations/GTEST_INSTALL        \
../
```

__Example 2:__ Specify the install directory, force an internal build NLOPT, and specify a url containing the NLOPT source code:
```
cmake                                                  \
  -DCMAKE_INSTALL_PREFIX=~/Installations/MUQ_INSTALL   \
  -DMUQ_FORCE_INTERNAL_NLOPT=ON                        \
  -DNLOPT_EXTERNAL_SOURCE=https://github.com/stevengj/nlopt/archive/v2.6.2.tar.gz \
../
```

__Example 3:__ Specify the install directory, enable MPI, use GTest, and specify the MPI compilers:
```
cmake                                                  \
  -DCMAKE_INSTALL_PREFIX=~/Installations/MUQ_INSTALL   \
  -DMUQ_USE_MPI=ON                                     \
  -DCMAKE_C_COMPILER=mpicc                             \
  -DCMAKE_CXX_COMPILER=mpic++                          \
  -DMUQ_USE_GTEST=ON                                   \
  -DMUQ_GTEST_DIR=~/Installations/GTEST_INSTALL        \
../
```

### Compile Groups
On a high level, MUQ is organized into separate libraries (e.g., muqModeling, muqApproximation). However, users have more granular control over what components of MUQ are compiled through our "compile group" concept.   A compile group is simply a set of related c++ source files with common dependencies that can be enabled or disabled during the CMake configuration.  If a group is disabled, it will not be compiled and its dependencies will not be used (or even searched for).   Below are some examples of enabling and disabling individual compile groups.

__Example 1:__ Turn off all compile groups except those needed by the HDF5 wrapper.  This will result in a single library `muqUtilities` that only contains MUQ's HDF5 wrapper.   The `MUQ_ENABLEGROUP_DEFAULT` option is turned off, which means that no compile groups will be included by default.  Then the HDF5 group is turned on with the `MUQ_ENABLEGROUP_<group name>` option.
```
cmake                                                  \
  -DCMAKE_INSTALL_PREFIX=~/Installations/MUQ_INSTALL   \
  -DMUQ_ENABLEGROUP_DEFAULT=OFF                        \
  -DMUQ_ENABLEGROUP_UTILITIES_HDF5=ON                  \
../
```

__Example 2:__ Use the default behavior of enabling all compile groups, but turn off the ODE group, which depends on Sundials.   With this configuration SUNDIALS is not required by MUQ and the `summary.log` file should show that SUNDIALS is "Not required for selected compile groups."
```
cmake                                                  \
  -DCMAKE_INSTALL_PREFIX=~/Installations/MUQ_INSTALL   \
  -DMUQ_ENABLEGROUP_MODELING_ODE=OFF                   \
../
```

## Compiling
CMake will generate a makefile that can then be used to compile MUQ in the usual fashion:
```
make -j4
make install
```

## Testing
If MUQ was configured with gtest (e.g., `MUQ_USE_GTEST` was set to `ON`), then compiling MUQ will produce a test executable called `RunAllTests`.   This executable can be run from the `build` directory using
```
./RunAllTests
```
GTest also provides functionality for runnsing a subset of the tests.  The following comman for example, will run all tests with "MCMC" in the name:
```
./RunAllTests --gtest_filter=*MCMC*
```
See the [GoogleTest](https://github.com/google/googletest/blob/master/googletest/docs/advanced.md#running-a-subset-of-the-tests) documentation for more details.
