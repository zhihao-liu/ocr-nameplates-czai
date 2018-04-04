cmake_minimum_required(VERSION 2.8)
project(cuizhou_ocr)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package(Boost REQUIRED)
## need to select your own opencv version
find_package(OpenCV 2.4.8 REQUIRED)

include_directories(
        /usr/include
        ${PROJECT_SOURCE_DIR}/include
        ${OpenCV_INCLUDE_DIR}
        ${PROJECT_SOURCE_DIR}/3rdparty/caffe/include
        ${PROJECT_SOURCE_DIR}/include/CuizhouOcr
        ${PROJECT_SOURCE_DIR}/include/PvaDetector
        ${PROJECT_SOURCE_DIR}/include/KCF
        ${PROJECT_SOURCE_DIR}/include/HungarianAlg
        /usr/local/cuda-8.0/include
        )

link_directories(
        ${PROJECT_SOURCE_DIR}/3rdparty/caffe/lib
        /usr/local/cuda-8.0/lib64
)

add_subdirectory(src)
add_subdirectory(executable)