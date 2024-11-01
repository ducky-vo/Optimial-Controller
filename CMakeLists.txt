cmake_minimum_required(VERSION 3.2)
project(OPTIMAL_CONTROLLER LANGUAGES CXX C)

# Set C++17.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find required packages.
find_package(GTest REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Threads REQUIRED)

# Define directories for source files.
set(SOURCE_DIR "source")
set(INCLUDE_DIR "include")
set(UTILITIES_DIR "utilities")
set(EXAMPLE_DIR "example")
set(TEST_DIR "test")

# Collect all source files.
file(GLOB SOURCES "${SOURCE_DIR}/*.cpp")

# Include directories.
include_directories(
  ${INCLUDE_DIR}
  ${UTILITIES_DIR}
  ${EIGEN3_INCLUDE_DIR}
)

# Build shared library.
add_library(${PROJECT_NAME}_lib SHARED ${SOURCES})
target_link_libraries(${PROJECT_NAME}_lib Eigen3::Eigen)

# Helper function to simplify example targets.
function(add_example EXAMPLE_NAME FILE_PATH)
  add_executable(${PROJECT_NAME}_${EXAMPLE_NAME} ${FILE_PATH})
  target_link_libraries(${PROJECT_NAME}_${EXAMPLE_NAME} ${PROJECT_NAME}_lib)
endfunction()

# Add examples.
add_example(FuzzyLogic ${EXAMPLE_DIR}/FuzzyLogic_Example.cpp)
add_example(LQR ${EXAMPLE_DIR}/LQR_Example.cpp)
add_example(LQG ${EXAMPLE_DIR}/LQG_Example.cpp)
add_example(MPC ${EXAMPLE_DIR}/MPC_Example.cpp)
add_example(PID ${EXAMPLE_DIR}/PID_Example.cpp)

# Add unit test.
add_executable(${PROJECT_NAME}_PID_Test ${TEST_DIR}/PID_Test.cpp)
target_link_libraries(${PROJECT_NAME}_PID_Test
  ${GTEST_LIBRARIES}
  ${PROJECT_NAME}_lib
  pthread
)

# Install targets.
install(TARGETS ${PROJECT_NAME}_LQR ${PROJECT_NAME}_LQG ${PROJECT_NAME}_MPC ${PROJECT_NAME}_PID
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

# Install headers.
install(DIRECTORY ${INCLUDE_DIR}/
  DESTINATION /usr/local/include
  FILES_MATCHING PATTERN "*.hpp"
)
