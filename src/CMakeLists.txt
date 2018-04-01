aux_source_directory(PvaDetector PVASRCS)
message(${PVASRCS})
add_library(pva SHARED ${PVASRCS})