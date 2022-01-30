# MIT License
# Copyright (c) 2018-Today Michele Adduci <adduci@tutanota.com>
#
# Project-related instructions

# Activate Testing, if given
if(BUILD_TESTING)
  enable_testing()
endif()

#####################################
# Define Targets
#####################################
add_subdirectory(mllib)
add_subdirectory(integration)

#####################################
# Define Install Targets
#####################################
install(TARGETS
  mllib
  integration
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  PUBLIC_HEADER DESTINATION include
)

install(
  FILES
    ${CMAKE_SOURCE_DIR}/project/mllib/include/linear_model.h
    ${CMAKE_BINARY_DIR}/exports/linear_model_export.h
  DESTINATION
    include/mllib)

install(FILES
  ${CMAKE_SOURCE_DIR}/LICENSE
DESTINATION .)