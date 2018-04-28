set(CAFFE_DEPENDENCIES boost_system ${OpenCV_LIBS} caffe glog)

aux_source_directory(mlmodel MLMODEL_SRC)
add_library(mlmodel SHARED ${MLMODEL_SRC} mlmodel/mlmodel.cpp)
target_link_libraries(mlmodel ${CAFFE_DEPENDENCIES})

aux_source_directory(cuizhou_ocr CUIZHOUOCR_SRCS)
aux_source_directory(cuizhou_ocr/data_utils CUIZHOUOCR_UTILS_SRCS)
aux_source_directory(cuizhou_ocr/ocr_aux CUIZHOUOCR_OCRAUX_SRCS)
aux_source_directory(cuizhou_ocr/ocr_implementation CUIZHOUOCR_OCRIMPLEMENTATION_SRCS)
add_library(cuizhou_ocr SHARED ${CUIZHOUOCR_SRCS} ${CUIZHOUOCR_UTILS_SRCS} ${CUIZHOUOCR_OCRAUX_SRCS} ${CUIZHOUOCR_OCRIMPLEMENTATION_SRCS} cuizhou_ocr/ocr_implementation/ocr_nameplate_volkswagen.cpp ../include/cuizhou_ocr/ocr_implementation/ocr_nameplate_volkswagen.h cuizhou_ocr/ocr_aux/ocr_detection.cpp cuizhou_ocr/ocr_aux/keyvalue_detection.cpp ../include/mlmodel/mlmodel.h)
target_link_libraries(cuizhou_ocr ${CAFFE_DEPENDENCIES} mlmodel)