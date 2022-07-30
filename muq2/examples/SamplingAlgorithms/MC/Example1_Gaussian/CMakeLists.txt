cmake_minimum_required (VERSION 3.10)

project(MCGaussian)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR) # Only look for MUQ when building example as standalone project
  find_package(MUQ REQUIRED)

  include_directories(${MUQ_INCLUDE_DIRS})

  set(CMAKE_CXX_FLAGS ${MUQ_CXX_FLAGS})
  set(CMAKE_CXX_COMPILER ${MUQ_CXX_COMPILER})
endif()


add_executable(MonteCarlo MonteCarlo.cpp)
target_link_libraries(MonteCarlo ${MUQ_LIBRARIES} ${MUQ_LINK_LIBRARIES})

add_executable(MultilevelMonteCarlo MultilevelMonteCarlo.cpp)
target_link_libraries(MultilevelMonteCarlo ${MUQ_LIBRARIES} ${MUQ_LINK_LIBRARIES})

if (MUQ_HAS_PARCER)
  add_executable(ParallelMultilevelMonteCarlo ParallelMultilevelMonteCarlo.cpp)
  target_link_libraries(ParallelMultilevelMonteCarlo ${MUQ_LIBRARIES} ${MUQ_LINK_LIBRARIES})
endif()
