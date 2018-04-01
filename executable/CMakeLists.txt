add_executable(demo demo.cpp)
set(Boost_LIBS boost_system boost_filesystem)
target_link_libraries(demo ${Boost_LIBS} ${OpenCV_LIBS} pva caffe)