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

# Utility rule file for sphinx.

# Include any custom commands dependencies for this target.
include doc/sphinx/CMakeFiles/sphinx.dir/compiler_depend.make

# Include the progress variables for this target.
include doc/sphinx/CMakeFiles/sphinx.dir/progress.make

doc/sphinx/CMakeFiles/sphinx:
	cd /home/anne/Masterarbeit/duneuro/build-release/duneuro-py/doc/sphinx && /usr/bin/sphinx-build -b html /home/anne/Masterarbeit/duneuro/build-release/duneuro-py/doc/sphinx /home/anne/Masterarbeit/duneuro/build-release/duneuro-py/doc/sphinx/html

sphinx: doc/sphinx/CMakeFiles/sphinx
sphinx: doc/sphinx/CMakeFiles/sphinx.dir/build.make
.PHONY : sphinx

# Rule to build all files generated by this target.
doc/sphinx/CMakeFiles/sphinx.dir/build: sphinx
.PHONY : doc/sphinx/CMakeFiles/sphinx.dir/build

doc/sphinx/CMakeFiles/sphinx.dir/clean:
	cd /home/anne/Masterarbeit/duneuro/build-release/duneuro-py/doc/sphinx && $(CMAKE_COMMAND) -P CMakeFiles/sphinx.dir/cmake_clean.cmake
.PHONY : doc/sphinx/CMakeFiles/sphinx.dir/clean

doc/sphinx/CMakeFiles/sphinx.dir/depend:
	cd /home/anne/Masterarbeit/duneuro/build-release/duneuro-py && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/anne/Masterarbeit/duneuro/duneuro-py /home/anne/Masterarbeit/duneuro/duneuro-py/doc/sphinx /home/anne/Masterarbeit/duneuro/build-release/duneuro-py /home/anne/Masterarbeit/duneuro/build-release/duneuro-py/doc/sphinx /home/anne/Masterarbeit/duneuro/build-release/duneuro-py/doc/sphinx/CMakeFiles/sphinx.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : doc/sphinx/CMakeFiles/sphinx.dir/depend

