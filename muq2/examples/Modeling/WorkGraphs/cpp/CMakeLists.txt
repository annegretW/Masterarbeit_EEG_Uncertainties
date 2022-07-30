cmake_minimum_required (VERSION 3.10)

project(SimpleWorkGraph)

set (CMAKE_CXX_STANDARD 17)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR) # Only look for MUQ when building example as standalone project
  find_package(MUQ REQUIRED)
  include_directories(${MUQ_INCLUDE_DIRS})
  message("INCLUDE DIRS: ${MUQ_INCLUDE_DIRS}")
endif()

add_executable(SimpleWorkGraph SimpleWorkGraph.cpp)
target_link_libraries(SimpleWorkGraph ${MUQ_LIBRARIES} ${MUQ_LINK_LIBRARIES})

add_executable(SplitSum SplitSumWorkGraph.cpp)
target_link_libraries(SplitSum ${MUQ_LIBRARIES} ${MUQ_LINK_LIBRARIES})
