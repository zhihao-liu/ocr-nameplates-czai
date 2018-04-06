//
// Created by Zhihao Liu on 18-4-4.
//

#include "OcrNameplates.h"


using namespace cuizhou;

std::string const OcrNameplates::CLASSNAME_VIN = "Vin";

OcrNameplates::~OcrNameplates() = default;

OcrNameplates::OcrNameplates() = default;

InfoTable const& OcrNameplates::getResult() const {
    return _result;
}