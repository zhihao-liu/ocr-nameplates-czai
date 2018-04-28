//
// Created by Zhihao Liu on 4/26/18.
//

#ifndef CUIZHOU_OCR_CV_EXTENSION_H
#define CUIZHOU_OCR_CV_EXTENSION_H

#include <opencv2/core/core.hpp>
#include "data_utils/perspective_transform.h"


namespace cz {

cv::Mat imgResizeAndFill(cv::Mat const& img, int newWidth, int newHeight, PerspectiveTransform* pForwardTransform = nullptr);
cv::Mat imgResizeAndFill(cv::Mat const& img, cv::Size const& newSize, PerspectiveTransform* pForwardTransform = nullptr);
cv::Mat imgRotate(cv::Mat const& img, double angleInDegree);

cv::Rect extent(cv::Mat const& img);

int xMid(cv::Rect const& rect);
int yMid(cv::Rect const& rect);

//int computeXOverlap(cv::Rect const& rect1, cv::Rect const& rect2);
//int computeYOverlap(cv::Rect const& rect1, cv::Rect const& rect2);
//int computeAreaIntersection(cv::Rect const& rect1, cv::Rect const& rect2);
float computeIou(cv::Rect const& rect1, cv::Rect const& rect2);
int computeSpacing(cv::Rect const& rect1, cv::Rect const& rect2);

} // end namespace cz

#endif //CUIZHOU_OCR_CV_EXTENSION_H
