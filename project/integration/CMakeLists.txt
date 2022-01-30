# MIT License
# Copyright (c) 2018-Today Michele Adduci <adduci@tutanota.com>

find_package(Eigen3 3.4 REQUIRED NO_MODULE)

#####################################
# Define Target Information
#####################################
set(TARGET_NAME integration)
set(TARGET_INCLUDE_FOLDER ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(TARGET_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/integration_test.cpp)

#####################################
# Support IDE Visualization
#####################################
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${TARGET_SOURCES})

#####################################
# Define Target according to options
#####################################
add_executable(${TARGET_NAME} ${TARGET_SOURCES})

# include all the necessary directories
target_include_directories(${TARGET_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/exports>
  PRIVATE
    ${TARGET_INCLUDE_FOLDER}
    ${CMAKE_CURRENT_SOURCE_DIR}/../mllib/include
    # Any other extra include required
)

# link against hellolib
target_link_libraries(${TARGET_NAME} PUBLIC mllib Eigen3::Eigen)
