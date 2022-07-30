if(NOT test_localized_subtraction_eeg_FOUND)
# Whether this module is installed or not
set(test_localized_subtraction_eeg_INSTALLED ON)

# Settings specific to the module

# Package initialization

####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was test_localized_subtraction_eeg-config.cmake.in                            ########

get_filename_component(PACKAGE_PREFIX_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../" ABSOLUTE)

macro(set_and_check _var _file)
  set(${_var} "${_file}")
  if(NOT EXISTS "${_file}")
    message(FATAL_ERROR "File or directory ${_file} referenced by variable ${_var} does not exist !")
  endif()
endmacro()

macro(check_required_components _NAME)
  foreach(comp ${${_NAME}_FIND_COMPONENTS})
    if(NOT ${_NAME}_${comp}_FOUND)
      if(${_NAME}_FIND_REQUIRED_${comp})
        set(${_NAME}_FOUND FALSE)
      endif()
    endif()
  endforeach()
endmacro()

####################################################################################

#report other information
set_and_check(test_localized_subtraction_eeg_PREFIX "${PACKAGE_PREFIX_DIR}")
set_and_check(test_localized_subtraction_eeg_INCLUDE_DIRS "${PACKAGE_PREFIX_DIR}/include")
set(test_localized_subtraction_eeg_CXX_FLAGS "-std=c++17 -O3 -std=c++17 -march=native")
set(test_localized_subtraction_eeg_CXX_FLAGS_DEBUG "-g")
set(test_localized_subtraction_eeg_CXX_FLAGS_MINSIZEREL "-Os -DNDEBUG")
set(test_localized_subtraction_eeg_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(test_localized_subtraction_eeg_CXX_FLAGS_RELWITHDEBINFO "-O2 -g -DNDEBUG")
set(test_localized_subtraction_eeg_DEPENDS "duneuro")
set(test_localized_subtraction_eeg_SUGGESTS "")
set(test_localized_subtraction_eeg_MODULE_PATH "${PACKAGE_PREFIX_DIR}/share/dune/cmake/modules")
set(test_localized_subtraction_eeg_LIBRARIES "")

# Lines that are set by the CMake build system via the variable DUNE_CUSTOM_PKG_CONFIG_SECTION


#import the target
if(test_localized_subtraction_eeg_LIBRARIES)
  get_filename_component(_dir "${CMAKE_CURRENT_LIST_FILE}" PATH)
  include("${_dir}/test_localized_subtraction_eeg-targets.cmake")
endif()
endif()
