cmake_minimum_required(VERSION 3.10)

project(BlockOperations)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR) # Only look for MUQ when building example as standalone project
  find_package(MUQ REQUIRED)

  set(CMAKE_CXX_COMPILER ${MUQ_CXX_COMPILER})
  set(CMAKE_CXX_FLAGS ${MUQ_CXX_FLAGS})

  include_directories(${MUQ_INCLUDE_DIRS})
endif()


add_executable(BlockOperations BlockOperations.cpp)
target_link_libraries(BlockOperations ${MUQ_LIBRARIES} ${MUQ_LINK_LIBRARIES})
