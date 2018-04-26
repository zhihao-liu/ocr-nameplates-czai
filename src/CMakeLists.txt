set(CAFFE_DEPENDENCIES boost_system ${OpenCV_LIBS} caffe glog)

aux_source_directory(mlmodel MLMODEL_SRC)
add_library(mlmodel SHARED ${MLMODEL_SRC})
target_link_libraries(mlmodel ${CAFFE_DEPENDENCIES})

aux_source_directory(cuizhou_ocr CUIZHOUOCR_SRCS)
aux_source_directory(cuizhou_ocr/utils CUIZHOUOCR_UTILS_SRCS)
add_library(cuizhou_ocr SHARED ${CUIZHOUOCR_SRCS} ${CUIZHOUOCR_UTILS_SRCS} cuizhou_ocr/utils/data_processing.cpp)
target_link_libraries(cuizhou_ocr ${CAFFE_DEPENDENCIES} mlmodel)