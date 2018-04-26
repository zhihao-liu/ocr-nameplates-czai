//
// Created by Zhihao Liu on 4/26/18.
//

#ifndef CUIZHOU_OCR_OCR_INTERFACE_H
#define CUIZHOU_OCR_OCR_INTERFACE_H

#include <memory>
#include "ocr_implementation/ocr_handler.h"
#include "ocr_implementation/ocr_nameplate_alfa.h"

namespace cuizhou {

enum class OcrType {
    NAMEPLATE_ALFA = 0
};

class OcrInterface {
public:
    ~OcrInterface() = default;
    OcrInterface() = delete;

    template<typename... Args>
    OcrInterface(OcrType type, Args&&... args);

    void inputImage(cv::Mat const& img);
    void processImage();

    cv::Mat const& image() const;
    cv::Mat drawResult() const;
    std::string getResultAsString() const;

private:
    std::shared_ptr<OcrHandler> ocrHandler_;
};

template<typename... Args>
OcrInterface::OcrInterface(OcrType type, Args&&... args) {
    switch(type) {
        case OcrType::NAMEPLATE_ALFA: {
            ocrHandler_ = std::make_shared<OcrNameplatesAlfa>(std::forward<Args>(args)...);
        } break;

        default: throw std::invalid_argument("Specified type for OcrInterface not supported.");
    }
}

} // end namespace cuizhou

#endif //CUIZHOU_OCR_OCR_INTERFACE_H
