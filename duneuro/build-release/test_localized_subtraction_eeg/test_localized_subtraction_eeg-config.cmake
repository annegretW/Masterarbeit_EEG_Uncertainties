if(NOT test_localized_subtraction_eeg_FOUND)
# Whether this module is installed or not
set(test_localized_subtraction_eeg_INSTALLED OFF)

# Settings specific to the module

# Package initialization
# Set prefix to source dir
set(PACKAGE_PREFIX_DIR /home/anne/Masterarbeit/duneuro/test_localized_subtraction_eeg)
macro(set_and_check _var _file)
  set(${_var} "${_file}")
  if(NOT EXISTS "${_file}")
    message(FATAL_ERROR "File or directory ${_file} referenced by variable ${_var} does not exist !")
  endif()
endmacro()

#report other information
set_and_check(test_localized_subtraction_eeg_PREFIX "${PACKAGE_PREFIX_DIR}")
set_and_check(test_localized_subtraction_eeg_INCLUDE_DIRS "/home/anne/Masterarbeit/duneuro/test_localized_subtraction_eeg")
set(test_localized_subtraction_eeg_CXX_FLAGS "-std=c++17 -O3 -std=c++17 -march=native")
set(test_localized_subtraction_eeg_CXX_FLAGS_DEBUG "-g")
set(test_localized_subtraction_eeg_CXX_FLAGS_MINSIZEREL "-Os -DNDEBUG")
set(test_localized_subtraction_eeg_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(test_localized_subtraction_eeg_CXX_FLAGS_RELWITHDEBINFO "-O2 -g -DNDEBUG")
set(test_localized_subtraction_eeg_DEPENDS "duneuro")
set(test_localized_subtraction_eeg_SUGGESTS "")
set(test_localized_subtraction_eeg_MODULE_PATH "/home/anne/Masterarbeit/duneuro/test_localized_subtraction_eeg/cmake/modules")
set(test_localized_subtraction_eeg_LIBRARIES "")

# Lines that are set by the CMake build system via the variable DUNE_CUSTOM_PKG_CONFIG_SECTION


#import the target
if(test_localized_subtraction_eeg_LIBRARIES)
  get_filename_component(_dir "${CMAKE_CURRENT_LIST_FILE}" PATH)
  include("${_dir}/test_localized_subtraction_eeg-targets.cmake")
endif()
endif()
