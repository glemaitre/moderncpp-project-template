# MIT License
# Copyright (c) 2018-Today Michele Adduci <adduci@tutanota.com>

find_package(Eigen3 3.4 REQUIRED NO_MODULE)
find_package(BLAS)
find_package(LAPACK)

#####################################
# Define Target Information
#####################################
set(TARGET_NAME mllib)
set(TARGET_INCLUDE_FOLDER ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(TARGET_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/linear_model.cpp)

#####################################
# Support IDE Visualization
#####################################
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${TARGET_SOURCES})

#####################################
# Define Target according to options
#####################################
if(BUILD_SHARED_LIBS)
  add_library(${TARGET_NAME} SHARED ${TARGET_SOURCES})
else()
  add_library(${TARGET_NAME} STATIC ${TARGET_SOURCES})
endif()

##########################
# Define Target Properties
##########################

# generate an export header to expose symbols when using the library
generate_export_header(${TARGET_NAME} EXPORT_FILE_NAME ${CMAKE_BINARY_DIR}/exports/${TARGET_NAME}_export.h)

# add target properties such as debug postfix, MSVC Runtime and library version
set_target_properties(${TARGET_NAME} PROPERTIES
                                      DEBUG_POSTFIX "-d"
                                      MSVC_RUNTIME_LIBRARY "${MSVC_RUNTIME_TYPE}"
                                      VERSION ${CMAKE_PROJECT_VERSION}
                                      SOVERSION ${CMAKE_PROJECT_VERSION})

# set target compile options as defined in the cmake/compiler_options.cmake Module
target_compile_options(${TARGET_NAME} PRIVATE ${compiler_options})

# set target link options as defined in the cmake/compiler_options.cmake Module
target_link_options(${TARGET_NAME} PRIVATE ${linker_flags})

# include all the necessary directories
target_include_directories(${TARGET_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/exports>
  PRIVATE
    ${TARGET_INCLUDE_FOLDER}
    # Any other extra include required
    ${EIGEN3_INCLUDE_DIR}
    ${BLAS_INCLUDE_DIRS}
    ${LAPACK_INCLUDE_DIRS}
)

add_definitions(-DEIGEN_USE_BLAS -DEIGEN_USE_LAPACKE)

#######################
# Setup the Google Test
#######################

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/609281088cfefc76f9d0ce82e1ff6c30cc3591e5.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

#########################
# Create binary for tests
#########################

enable_testing()

add_executable(
  linear_model_test
  ${CMAKE_CURRENT_SOURCE_DIR}/test/linear_model.cpp
)

# include all the necessary directories
target_include_directories(${TARGET_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/exports>
  PRIVATE
    ${TARGET_INCLUDE_FOLDER}
    # Any other extra include required
)

target_link_libraries(linear_model_test PUBLIC mllib Eigen3::Eigen lapack blas gtest_main)

include(GoogleTest)
gtest_discover_tests(linear_model_test)
