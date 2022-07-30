if(NOT duneuro-py_FOUND)
# Whether this module is installed or not
set(duneuro-py_INSTALLED OFF)

# Settings specific to the module

# Package initialization
# Set prefix to source dir
set(PACKAGE_PREFIX_DIR /home/anne/Masterarbeit/duneuro/duneuro-py)
macro(set_and_check _var _file)
  set(${_var} "${_file}")
  if(NOT EXISTS "${_file}")
    message(FATAL_ERROR "File or directory ${_file} referenced by variable ${_var} does not exist !")
  endif()
endmacro()

#report other information
set_and_check(duneuro-py_PREFIX "${PACKAGE_PREFIX_DIR}")
set_and_check(duneuro-py_INCLUDE_DIRS "/home/anne/Masterarbeit/duneuro/duneuro-py")
set(duneuro-py_CXX_FLAGS "-std=c++17 -O3 -std=c++17 -march=native")
set(duneuro-py_CXX_FLAGS_DEBUG "-g")
set(duneuro-py_CXX_FLAGS_MINSIZEREL "-Os -DNDEBUG")
set(duneuro-py_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(duneuro-py_CXX_FLAGS_RELWITHDEBINFO "-O2 -g -DNDEBUG")
set(duneuro-py_DEPENDS "dune-common (>= 2.5);duneuro (>= 2.5)")
set(duneuro-py_SUGGESTS "")
set(duneuro-py_MODULE_PATH "/home/anne/Masterarbeit/duneuro/duneuro-py/cmake/modules")
set(duneuro-py_LIBRARIES "")

# Lines that are set by the CMake build system via the variable DUNE_CUSTOM_PKG_CONFIG_SECTION


#import the target
if(duneuro-py_LIBRARIES)
  get_filename_component(_dir "${CMAKE_CURRENT_LIST_FILE}" PATH)
  include("${_dir}/duneuro-py-targets.cmake")
endif()
endif()
