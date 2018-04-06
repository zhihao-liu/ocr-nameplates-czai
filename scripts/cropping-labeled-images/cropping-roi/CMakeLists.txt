cmake_minimum_required(VERSION 3.9)
project(cropping_roi)

set(CMAKE_CXX_STANDARD 17)

add_executable(cropping_roi main.cpp)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(cropping_roi ${OpenCV_LIBS})