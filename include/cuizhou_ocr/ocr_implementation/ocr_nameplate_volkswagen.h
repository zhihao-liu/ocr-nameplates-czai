//
// Created by Zhihao Liu on 4/27/18.
//

#ifndef CUIZHOU_OCR_OCR_NAMEPLATE_VOLKSWAGEN_H
#define CUIZHOU_OCR_OCR_NAMEPLATE_VOLKSWAGEN_H

#include "detector.h"
#include "ocr_implementation/ocr_nameplate.h"
#include "ocr_aux/collage.hpp"


namespace cz {

class OcrNameplateVolkswagen final : public OcrNameplate {
public:
    ~OcrNameplateVolkswagen() override;
    OcrNameplateVolkswagen() = delete;

    explicit OcrNameplateVolkswagen(Detector detectorValues);

    virtual void processImage(ShowProgress const& showProgress = ShowProgress()) override;

private:
    static EnumHashMap<NameplateField, int> const VALUE_LENGTH;

    Detector detectorValues_;
};

} // end namespace cz


#endif //CUIZHOU_OCR_OCR_NAMEPLATE_VOLKSWAGEN_H
