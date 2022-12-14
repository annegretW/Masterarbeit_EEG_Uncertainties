include(ExternalProject)

if(NOT MUQ_INTERNAL_NLOPT_VERSION)
  set(MUQ_INTERNAL_NLOPT_VERSION "2.4.2")
endif()

if(NOT NLOPT_EXTERNAL_SOURCE)

  set(NLOPT_EXTERNAL_SOURCE http://ab-initio.mit.edu/nlopt/nlopt-${MUQ_INTERNAL_NLOPT_VERSION}.tar.gz)
  message(STATUS "Will download NLOPT from ${NLOPT_EXTERNAL_SOURCE} during compile.")

endif()

set(NLOPT_CFLAGS "")
if(CMAKE_OSX_ARCHITECTURES)
  set(NLOPT_CFLAGS "${NLOPT_CFLAGS}-arch arm64 -arch x86_64")
endif()

#set(NLOPT_CFLAGS "\"${NLOPT_CFLAGS}\"")


set(NLOPT_INSTALL_DIR ${CMAKE_INSTALL_PREFIX}/muq_external/)
ExternalProject_Add(
  NLOPT
    PREFIX ${CMAKE_CURRENT_BINARY_DIR}/external/nlopt
    URL ${NLOPT_EXTERNAL_SOURCE}
    CONFIGURE_COMMAND ${CMAKE_CURRENT_BINARY_DIR}/external/nlopt/src/NLOPT/configure  CC=${CMAKE_C_COMPILER} CXX=${CMAKE_CXX_COMPILER} ${HDF5_PARALLEL_FLAG} CFLAGS=${NLOPT_CFLAGS} --prefix=${NLOPT_INSTALL_DIR} --enable-shared --without-octave --without-matlab --without-python --without-guile
    BUILD_COMMAND $(MAKE) install
    BUILD_IN_SOURCE 1
    INSTALL_COMMAND ""
)


set_property( TARGET NLOPT PROPERTY FOLDER "Externals")

set(NLOPT_INCLUDE_DIRS "${NLOPT_INSTALL_DIR}include" )
set(NLOPT_LIBRARIES ${NLOPT_INSTALL_DIR}/lib/${CMAKE_SHARED_LIBRARY_PREFIX}nlopt${CMAKE_SHARED_LIBRARY_SUFFIX})
