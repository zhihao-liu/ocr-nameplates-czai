//
// Created by Zhihao Liu on 4/26/18.
//

#include "ocr_interface.h"
#include "ocr_implementation/ocr_nameplate_alfaromeo.h"
#include "ocr_implementation/ocr_nameplate_volkswagen.h"

namespace cz {

OcrInterface::~OcrInterface() = default;

OcrInterface::OcrInterface(OcrType type, std::initializer_list<std::reference_wrapper<MlModel const>> refModels) {
    std::vector<std::reference_wrapper<MlModel const>> models(refModels);
    switch (type) {
        case OcrType::NAMEPLATE_ALFAROMEO: {
            std::invalid_argument ex("Invalid arguments for initialization of 'OcrNameplateAlfaRomeo'.");
            if (models.size() != 4) throw ex;

            Detector const& detectorKeys = dynamic_cast<Detector const&>(models.at(0).get());
            Detector const& detectorValueVin = dynamic_cast<Detector const&>(models.at(1).get());
            Detector const& detectorValuesStitched = dynamic_cast<Detector const&>(models.at(2).get());
            Classifier const& classifierChars = dynamic_cast<Classifier const&>(models.at(3).get());

            ocrHandler_ = std::make_shared<OcrNameplateAlfaRomeo>(detectorKeys, detectorValueVin, detectorValuesStitched, classifierChars);
        } break;

        case OcrType::NAMEPLATE_VOLKSWAGEN: {
            std::invalid_argument ex("Invalid arguments for initialization of 'OcrNameplateVolkswagen'.");
            if (models.size() != 1) throw ex;

            Detector const& detectorValues = dynamic_cast<Detector const&>(models.at(0).get());

            ocrHandler_ = std::make_shared<OcrNameplateVolkswagen>(detectorValues);
        } break;

        default: throw std::invalid_argument("Type unsupported for OCR Interface.");
    }
}

void OcrInterface::importImage(cv::Mat const& img) {
    ocrHandler_->importImage(img);
}

void OcrInterface::setImageSource(cv::Mat const& img) {
    ocrHandler_->setImageSource(img);
}

void OcrInterface::processImage(OcrHandler::ShowProgress const& showProgress) {
    ocrHandler_->processImage(showProgress);
}

cv::Mat const& OcrInterface::image() const {
    return ocrHandler_->image();
}

std::string OcrInterface::getResultAsString() const {
    return ocrHandler_->getResultAsString();
}

cv::Mat OcrInterface::drawResult() const {
    return ocrHandler_->drawResult();
}

} // end namespace cz