# MIT License
# Copyright (c) 2018-Today Michele Adduci <adduci@tutanota.com>
#
# hellolib target instructions

#####################################
# Define Target Information
#####################################
set(TARGET_NAME machine_learning)
set(TARGET_INCLUDE_FOLDER ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(
    TARGET_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hellolib.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/linear_model.cpp
)

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

# include all the necessary directories
target_include_directories(${TARGET_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/exports>
  PRIVATE
    ${TARGET_INCLUDE_FOLDER}
    # include OpenSSL directories if present
    $<$<BOOL:${OPENSSL_FOUND}>:${OPENSSL_INCLUDE_DIR}>
    # Any other extra include required
)

# if(BUILD_TESTING)
#   list(APPEND TEST_CASES
#     hellolib_test
#   )
#   foreach(TEST_CASE ${TEST_CASES})
#     add_executable(${TEST_CASE} ${CMAKE_CURRENT_SOURCE_DIR}/test/${TEST_CASE}.cpp)
#     target_link_libraries(${TEST_CASE} PRIVATE ${TARGET_NAME} doctest::doctest)
#     target_compile_options(${TEST_CASE} PRIVATE ${compiler_options})
#     target_compile_definitions(${TEST_CASE} PRIVATE ${compiler_definitions} $<$<BOOL:${OPENSSL_FOUND}>:WITH_OPENSSL>)
#     target_link_options(${TEST_CASE} PRIVATE ${linker_flags})

#     target_include_directories(${TEST_CASE}
#       PUBLIC
#         $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/exports>
#       PRIVATE
#         ${TARGET_INCLUDE_FOLDER}
#         # Any other extra include required
#     )

#     add_test(NAME ${TEST_CASE} COMMAND ${TEST_CASE})
#   endforeach(TEST_CASE ${TEST_CASES})
# endif()