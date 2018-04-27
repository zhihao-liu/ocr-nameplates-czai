//
// Created by Zhihao Liu on 4/26/18.
//

#ifndef CUIZHOU_OCR_OCR_INTERFACE_H
#define CUIZHOU_OCR_OCR_INTERFACE_H

#include <memory>
#include "mlmodel.h"
#include "ocr_implementation/ocr_handler.h"

namespace cuizhou {

enum class OcrType {
    NAMEPLATE_ALFAROMEO = 0,
    NAMEPLATE_VOLKSWAGEN
};

class OcrInterface {
public:
    ~OcrInterface() = default;
    OcrInterface() = delete;

    OcrInterface(OcrType type, std::initializer_list<std::reference_wrapper<MlModel const>> refModels);

    void importImage(cv::Mat const& img);
    void setImageSource(cv::Mat const& img);
    void processImage();

    cv::Mat const& image() const;
    cv::Mat drawResult() const;
    std::string getResultAsString() const;

private:
    std::shared_ptr<OcrHandler> ocrHandler_;
};

} // end namespace cuizhou

#endif //CUIZHOU_OCR_OCR_INTERFACE_H
