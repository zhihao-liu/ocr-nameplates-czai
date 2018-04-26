//
// Created by Zhihao Liu on 4/26/18.
//

#ifndef CUIZHOU_OCR_CV_EXTENSION_H
#define CUIZHOU_OCR_CV_EXTENSION_H

#include <opencv2/core/core.hpp>
#include "utils/perspective_transform.h"


namespace cuizhou {

cv::Mat imgResizeAndFill(cv::Mat const& img, int newWidth, int newHeight, PerspectiveTransform* pForwardTransform = nullptr);
cv::Mat imgResizeAndFill(cv::Mat const& img, cv::Size const& newSize, PerspectiveTransform* pForwardTransform = nullptr);

cv::Mat imgRotate(cv::Mat const& img, double angleInDegree);

int xMid(cv::Rect const& rect);
int yMid(cv::Rect const& rect);

int computeXOverlap(cv::Rect const& rect1, cv::Rect const& rect2);
int computeYOverlap(cv::Rect const& rect1, cv::Rect const& rect2);
int computeAreaIntersection(cv::Rect const& rect1, cv::Rect const& rect2);
float computeIou(cv::Rect const& rect1, cv::Rect const& rect2);
int computeSpacing(cv::Rect const& rect1, cv::Rect const& rect2);

template<typename Rect> Rect&& validateRoi(Rect&& roi, int width, int height);
template<typename Rect> Rect&& validateRoi(Rect&& roi, cv::Mat const& img);
template<typename Rect> Rect&& validateRoi(Rect&& roi, cv::Rect const& extent);


/* ------- Template Implementations ------- */

template<typename Rect>
Rect&& validateRoi(Rect&& roi, int width, int height) {
    roi.x = std::min(std::max(roi.x, 0), width - 1);
    roi.y = std::min(std::max(roi.y, 0), height - 1);
    roi.width = std::min(std::max(roi.width, 1), width - roi.x);
    roi.height = std::min(std::max(roi.height, 1), height - roi.y);

    return roi;
}

template<typename Rect>
Rect&& validateRoi(Rect&& roi, cv::Mat const& img) {
    // ensure the roi is within the extent of the image after adjustments
    return validateRoi(roi, img.cols, img.rows);
}

template<typename Rect>
Rect&& validateRoi(Rect&& roi, cv::Rect const& extent) {
    // ensure the roi is within the extent of the image after adjustments
    return validateRoi(roi, extent.width, extent.height);
}


} // end namespace cuizhou

#endif //CUIZHOU_OCR_CV_EXTENSION_H
