//
// Created by Zhihao Liu on 18-4-4.
//

#ifndef OCR_CUIZHOU_OCRHANDLER_H
#define OCR_CUIZHOU_OCRHANDLER_H

#include <opencv2/core/core.hpp>


namespace cuizhou {

class OcrHandler {
public:
    virtual ~OcrHandler() = default;
    void setImage(cv::Mat const& image);
    cv::Mat const& image() const;
    virtual void processImage() = 0;
    virtual cv::Mat drawResult() const = 0;

protected:
    cv::Mat image_;

    OcrHandler() = default;
};

} // end namespace cuizhou


#endif // OCR_CUIZHOU_OCRHANDLER_H
