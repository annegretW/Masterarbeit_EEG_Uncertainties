include(ExternalProject)


if(NOT PARCER_EXTERNAL_SOURCE)

  set(PARCER_EXTERNAL_SOURCE https://bitbucket.org/mituq/parcer/get/master.zip)
  message(STATUS "Will download PARCER from ${PARCER_EXTERNAL_SOURCE} during compile.")

endif()

set(PARCER_INSTALL_DIR ${CMAKE_INSTALL_PREFIX}/muq_external/)

ExternalProject_Add(
  PARCER
    PREFIX ${CMAKE_CURRENT_BINARY_DIR}/external/parcer
    URL ${PARCER_EXTERNAL_SOURCE}
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${PARCER_INSTALL_DIR} -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER} -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
    BUILD_COMMAND $(MAKE)
    INSTALL_COMMAND make install
)

set(PARCER_LIBRARIES ${PARCER_INSTALL_DIR}/lib/${CMAKE_SHARED_LIBRARY_PREFIX}parcer${CMAKE_SHARED_LIBRARY_SUFFIX})
set(PARCER_LIBRARY ${PARCER_LIBRARIES})

set_property( TARGET PARCER PROPERTY FOLDER "Externals")

set(PARCER_INCLUDE_DIRS ${PARCER_INSTALL_DIR}/include)
message(STATUS "Adding ${PARCER_INSTALL_DIR}/include for an ParCer include directory.")
