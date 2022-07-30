cmake_minimum_required (VERSION 3.10)

project(MultilevelMCMC_FLowModel)

set (CMAKE_CXX_STANDARD 17)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR) # Only look for MUQ when building example as standalone project
  find_package(MUQ REQUIRED HINTS ${MUQ_DIR})
  include_directories(${MUQ_INCLUDE_DIRS})
endif()

set(SOURCE
    MultilevelMCMC_FLowModel.cpp
    FlowModelComponents.cpp
)

add_executable(MultilevelMCMC_FLowModel ${SOURCE})
target_link_libraries(MultilevelMCMC_FLowModel ${MUQ_LIBRARIES} ${MUQ_LINK_LIBRARIES})
