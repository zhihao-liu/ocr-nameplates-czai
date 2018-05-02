//
// Created by Zhihao Liu on 4/27/18.
//

#include "ocr_implementation/ocr_nameplate_volkswagen.h"

namespace cz {

OcrNameplateVolkswagen::~OcrNameplateVolkswagen() = default;

OcrNameplateVolkswagen::OcrNameplateVolkswagen(Detector detectorValues)
        : detectorValues_(std::move(detectorValues)) {}

void OcrNameplateVolkswagen::processImage(ShowProgress const& showProgress) {
    detectorValues_.setThresh(0.1, 0.2);
    std::vector<Detection> dets = detectorValues_.detect(image_);
    Detector::drawBox(image_, dets);
}

} // end namespace cz