cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(histogram_equalization LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

# Find matplotlib package
find_package(Python COMPONENTS Development NumPy)

# Check if the package is found
if (Python_FOUND)
    include_directories(${Python_INCLUDE_DIRS})
    include_directories(${Python_NumPy_INCLUDE_DIRS})

    # source to build
    add_executable(histogram_equalization src/main.cpp)

    # Link against the required libraries
    target_link_libraries(histogram_equalization PRIVATE ${Python_LIBRARIES})
else()
    message(FATAL_ERROR "Matplotlib not found. Please make sure it is installed.")
endif()