# Install script for directory: /home/anne/Masterarbeit/duneuro/test_localized_subtraction_eeg

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  set(CMAKE_MODULE_PATH /home/anne/Masterarbeit/duneuro/test_localized_subtraction_eeg/cmake/modules;/home/anne/Masterarbeit/duneuro/duneuro/cmake/modules;/home/anne/Masterarbeit/duneuro/dune-pdelab/cmake/modules;/home/anne/Masterarbeit/duneuro/dune-functions/cmake/modules;/home/anne/Masterarbeit/duneuro/dune-alugrid/cmake/modules;/home/anne/Masterarbeit/duneuro/dune-localfunctions/cmake/modules;/home/anne/Masterarbeit/duneuro/dune-istl/cmake/modules;/home/anne/Masterarbeit/duneuro/dune-typetree/cmake/modules;/home/anne/Masterarbeit/duneuro/dune-grid/cmake/modules;/home/anne/Masterarbeit/duneuro/dune-uggrid/cmake/modules;/home/anne/Masterarbeit/duneuro/dune-geometry/cmake/modules;/home/anne/Masterarbeit/duneuro/dune-common/cmake/modules)
              set(DUNE_PYTHON_WHEELHOUSE /usr/local/share/dune/wheelhouse)
              include(DuneExecuteProcess)
              dune_execute_process(COMMAND "/usr/bin/cmake" --build . --target install_python --config $<CONFIG>)
              
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/dunecontrol/test_localized_subtraction_eeg" TYPE FILE FILES "/home/anne/Masterarbeit/duneuro/test_localized_subtraction_eeg/dune.module")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/test_localized_subtraction_eeg" TYPE FILE FILES
    "/home/anne/Masterarbeit/duneuro/build-release/test_localized_subtraction_eeg/cmake/pkg/test_localized_subtraction_eeg-config.cmake"
    "/home/anne/Masterarbeit/duneuro/build-release/test_localized_subtraction_eeg/test_localized_subtraction_eeg-config-version.cmake"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/test_localized_subtraction_eeg" TYPE FILE FILES "/home/anne/Masterarbeit/duneuro/test_localized_subtraction_eeg/config.h.cmake")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/pkgconfig" TYPE FILE FILES "/home/anne/Masterarbeit/duneuro/build-release/test_localized_subtraction_eeg/test_localized_subtraction_eeg.pc")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("/home/anne/Masterarbeit/duneuro/build-release/test_localized_subtraction_eeg/src/cmake_install.cmake")
  include("/home/anne/Masterarbeit/duneuro/build-release/test_localized_subtraction_eeg/dune/cmake_install.cmake")
  include("/home/anne/Masterarbeit/duneuro/build-release/test_localized_subtraction_eeg/doc/cmake_install.cmake")
  include("/home/anne/Masterarbeit/duneuro/build-release/test_localized_subtraction_eeg/cmake/modules/cmake_install.cmake")

endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "/home/anne/Masterarbeit/duneuro/build-release/test_localized_subtraction_eeg/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
