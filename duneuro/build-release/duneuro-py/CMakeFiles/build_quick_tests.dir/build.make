# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/anne/Masterarbeit/duneuro/duneuro-py

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/anne/Masterarbeit/duneuro/build-release/duneuro-py

# Utility rule file for build_quick_tests.

# Include any custom commands dependencies for this target.
include CMakeFiles/build_quick_tests.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/build_quick_tests.dir/progress.make

build_quick_tests: CMakeFiles/build_quick_tests.dir/build.make
.PHONY : build_quick_tests

# Rule to build all files generated by this target.
CMakeFiles/build_quick_tests.dir/build: build_quick_tests
.PHONY : CMakeFiles/build_quick_tests.dir/build

CMakeFiles/build_quick_tests.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/build_quick_tests.dir/cmake_clean.cmake
.PHONY : CMakeFiles/build_quick_tests.dir/clean

CMakeFiles/build_quick_tests.dir/depend:
	cd /home/anne/Masterarbeit/duneuro/build-release/duneuro-py && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/anne/Masterarbeit/duneuro/duneuro-py /home/anne/Masterarbeit/duneuro/duneuro-py /home/anne/Masterarbeit/duneuro/build-release/duneuro-py /home/anne/Masterarbeit/duneuro/build-release/duneuro-py /home/anne/Masterarbeit/duneuro/build-release/duneuro-py/CMakeFiles/build_quick_tests.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/build_quick_tests.dir/depend

