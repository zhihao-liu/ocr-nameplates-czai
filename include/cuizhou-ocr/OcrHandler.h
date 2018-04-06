//
// Created by Zhihao Liu on 18-4-4.
//

#ifndef OCR_CUIZHOU_OCRHANDLER_H
#define OCR_CUIZHOU_OCRHANDLER_H

#include "opencv2/core/core.hpp"


namespace cuizhou {
    class OcrHandler {
    public:
        virtual ~OcrHandler();
        void setImage(cv::Mat const& image);
        cv::Mat const& getImage() const;
        virtual void processImage() = 0;

    protected:
        cv::Mat _image;

        OcrHandler();
    };
}


#endif // OCR_CUIZHOU_OCRHANDLER_H
