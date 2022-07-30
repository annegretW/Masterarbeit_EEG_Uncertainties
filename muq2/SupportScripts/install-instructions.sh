# First, let's update our package manager and install some basic dependencies
apt update
apt install -y git cmake g++

# Optionally, install some advanced dependencies. muq will automatically download these if not found on your system.
apt install -y libhdf5-dev libboost-all-dev libeigen3-dev libsundials-dev libnlopt-dev


# Get muq from git repository
git clone https://bitbucket.org/mituq/muq2.git


# Let's compile! This is just the usual cmake build procedure.
cd muq2/build

# Here you can choose your install directory by setting CMAKE_INSTALL_PREFIX
cmake -DCMAKE_INSTALL_PREFIX=$PWD/install ..

make -j4

make install


# Now let's try out our shiny new muq install with an example!

make -j4 examples

cd examples/SamplingAlgorithms/MCMC/Example1_Gaussian/cpp

./GaussianSampling
