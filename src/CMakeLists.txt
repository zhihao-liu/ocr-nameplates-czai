aux_source_directory(PvaDetector PVASRCS)
add_library(pva_detector SHARED ${PVASRCS})

aux_source_directory(CuizhouOcr OCRSRCS)
add_library(cuizhou_ocr SHARED ${OCRSRCS} CuizhouOcr/OcrUtils.cpp ../include/CuizhouOcr/OcrUtils.hpp)
