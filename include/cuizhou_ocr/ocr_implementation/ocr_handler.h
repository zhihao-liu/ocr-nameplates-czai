//
// Created by Zhihao Liu on 18-4-4.
//

#ifndef OCR_CUIZHOU_OCRHANDLER_H
#define OCR_CUIZHOU_OCRHANDLER_H

#include <opencv2/core/core.hpp>


namespace cz {

class OcrHandler {
public:
    virtual ~OcrHandler();

    void importImage(cv::Mat const& image);
    void setImageSource(cv::Mat const& image);
    virtual void processImage() = 0;

    cv::Mat const& image() const;

    virtual cv::Mat drawResult() const = 0;
    virtual std::string getResultAsString() const = 0;

protected:
    cv::Mat image_;

    OcrHandler();
};

} // end namespace cz


#endif // OCR_CUIZHOU_OCRHANDLER_H
