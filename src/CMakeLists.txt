set(CAFFE_DEPENDENCIES boost_system ${OpenCV_LIBS} caffe glog)

aux_source_directory(pvadetector PVA_SRCS)
add_library(pvadetector SHARED ${PVA_SRCS})
target_link_libraries(pvadetector ${CAFFE_DEPENDENCIES})

aux_source_directory(googlenet GOOGLENET_SRC)
add_library(googlenet SHARED ${GOOGLENET_SRC})
target_link_libraries(googlenet ${CAFFE_DEPENDENCIES})

aux_source_directory(cuizhou_ocr CUIZHOUOCR_SRCS)
add_library(cuizhou_ocr SHARED ${CUIZHOUOCR_SRCS} ../include/cuizhou_ocr/classname_dict.hpp ../include/cuizhou_ocr/enum_hashmap.hpp cuizhou_ocr/collage.cpp ../include/cuizhou_ocr/collage.hpp)
target_link_libraries(cuizhou_ocr ${CAFFE_DEPENDENCIES} pvadetector googlenet)