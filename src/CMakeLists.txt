aux_source_directory(pvadetector PVA_SRCS)
add_library(pva_detector SHARED ${PVA_SRCS})

aux_source_directory(googlenet GOOGLENET_SRC)
add_library(googlenet SHARED ${GOOGLENET_SRC})

aux_source_directory(cuizhou-ocr CUIZHOUOCR_SRCS)
add_library(cuizhou_ocr SHARED ${CUIZHOUOCR_SRCS})