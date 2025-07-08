cmake_minimum_required(VERSION 3.15)
project(bts_algo VERSION 0.1.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set build type if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Add compiler flags
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# Add custom cmake modules path
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Find required packages
find_package(PkgConfig REQUIRED)
find_package(Threads REQUIRED)

# Find TensorFlow C API
find_package(TensorFlow REQUIRED)

# Find Boost (for asio)
find_package(Boost REQUIRED COMPONENTS system)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/cpp/include)
include_directories(${TensorFlow_INCLUDE_DIRS})

# Add subdirectories
add_subdirectory(cpp)

# Optional: Add testing if GoogleTest is found
find_package(GTest QUIET)
if(GTest_FOUND)
    enable_testing()
    add_subdirectory(tests/cpp)
endif()

# Print configuration
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "TensorFlow found: ${TensorFlow_FOUND}")
message(STATUS "TensorFlow include: ${TensorFlow_INCLUDE_DIRS}")
message(STATUS "TensorFlow libraries: ${TensorFlow_LIBRARIES}")
message(STATUS "Boost found: ${Boost_FOUND}")
message(STATUS "GoogleTest found: ${GTest_FOUND}") 