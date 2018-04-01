add_executable(demo demo.cpp)
target_link_libraries(demo ${OpenCV_LIBS} pva caffe glog boost_system)