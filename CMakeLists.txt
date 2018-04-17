cmake_minimum_required(VERSION 2.8)
project(cuizhou_ocr)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
#set(CMAKE_MACOSX_RPATH 1)

find_package(Boost REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(
        /usr/include
        ${PROJECT_SOURCE_DIR}/include
        ${OpenCV_INCLUDE_DIRS}
        ${PROJECT_SOURCE_DIR}/3rdparty/caffe/include
        ${PROJECT_SOURCE_DIR}/include/googlenet
        ${PROJECT_SOURCE_DIR}/include/cuizhou-ocr
        ${PROJECT_SOURCE_DIR}/include/pvadetector
        /usr/local/cuda/include
        )

link_directories(
        ${PROJECT_SOURCE_DIR}/3rdparty/caffe/lib
        /usr/local/cuda/lib
)

add_subdirectory(src)
add_subdirectory(executable)