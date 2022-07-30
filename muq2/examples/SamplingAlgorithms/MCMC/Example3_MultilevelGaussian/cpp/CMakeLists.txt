cmake_minimum_required (VERSION 3.10)

project(MultilevelGaussian)

set (CMAKE_CXX_STANDARD 17)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR) # Only look for MUQ when building example as standalone project
  find_package(MUQ REQUIRED)
  include_directories(${MUQ_INCLUDE_DIRS})
endif()

add_executable(BasicMultilevel MultilevelMCMC_BasicInterface.cpp)
target_link_libraries(BasicMultilevel ${MUQ_LIBRARIES} ${MUQ_LINK_LIBRARIES})

add_executable(AdvancedMultilevel MultilevelMCMC_AdvancedInterface.cpp)
target_link_libraries(AdvancedMultilevel ${MUQ_LIBRARIES} ${MUQ_LINK_LIBRARIES})


if (MUQ_HAS_PARCER)
  add_executable(ModelParallelMultilevelGaussianSampling ModelParallelMultilevelGaussianSampling.cpp)
  target_link_libraries(ModelParallelMultilevelGaussianSampling ${MUQ_LIBRARIES} ${MUQ_LINK_LIBRARIES})

  add_executable(FullParallelMultilevelGaussianSampling FullParallelMultilevelGaussianSampling.cpp)
  target_link_libraries(FullParallelMultilevelGaussianSampling ${MUQ_LIBRARIES} ${MUQ_LINK_LIBRARIES})

  add_executable(SubsamplingTestMultilevelGaussianSampling SubsamplingTestMultilevelGaussianSampling.cpp)
  target_link_libraries(SubsamplingTestMultilevelGaussianSampling ${MUQ_LIBRARIES} ${MUQ_LINK_LIBRARIES})
endif()
