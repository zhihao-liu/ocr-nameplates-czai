//
// Created by Zhihao Liu on 4/26/18.
//

#include <opencv2/imgproc/imgproc.hpp>
#include "utils/cv_extension.h"


namespace cuizhou {

cv::Mat imgResizeAndFill(cv::Mat const& img,
                         cv::Size const& newSize,
                         PerspectiveTransform* pForwardTransform) {
    return imgResizeAndFill(img, newSize.width, newSize.height, pForwardTransform);
}

// resize the image while preserve its width-height ratio
// save the transform parameters into pForwardTransform if it is non-null
cv::Mat imgResizeAndFill(cv::Mat const& img,
                                   int newWidth, int newHeight,
                                   PerspectiveTransform* pForwardTransform) {
    if (img.cols == newWidth && img.rows == newHeight) return img.clone();

    cv::Mat newImg(newHeight, newWidth, CV_8UC3, cv::Scalar(0, 0, 0));;
    float wScale = float(newWidth) / img.cols;
    float hScale = float(newHeight) / img.rows;

    float unifiedScale = wScale < hScale ? wScale : hScale;

    cv::Mat tempImg;
    cv::resize(img, tempImg, cv::Size(int(unifiedScale * img.cols), int(unifiedScale * img.rows)));

    int xShift = (newImg.cols - tempImg.cols) / 2;
    int yShift = (newImg.rows - tempImg.rows) / 2;
    tempImg.copyTo(newImg(cv::Rect(xShift, yShift, tempImg.cols, tempImg.rows)));

    if (pForwardTransform) {
        pForwardTransform->setOffset(xShift, yShift);
        pForwardTransform->setScale(unifiedScale);
    }

    return newImg;
}

cv::Mat imgRotate(cv::Mat const& img, double angleInDegree) {
    cv::Mat newImg;

    cv::Point2d pt(img.cols / 2.0, img.rows / 2.0);
    cv::Mat r = cv::getRotationMatrix2D(pt, angleInDegree, 1.0);
    cv::warpAffine(img, newImg, r, img.size());

    return newImg;
}

int xMid(cv::Rect const& rect) {
    return rect.x + int(round(rect.width / 2.0));
}

int yMid(cv::Rect const& rect) {
    return rect.y + int(round(rect.height / 2.0));
}

int computeXOverlap(cv::Rect const& rect1, cv::Rect const& rect2) {
    int xMin = std::min(rect1.x, rect2.x);
    int xMax = std::max(rect1.x + rect1.width, rect2.x + rect2.width);
    int xOverlap = (rect1.width + rect2.width) - (xMax - xMin);

    return xOverlap <= 0 ? 0 : xOverlap;
}

int computeYOverlap(cv::Rect const& rect1, cv::Rect const& rect2) {
    int yMin = std::min(rect1.y, rect2.y);
    int yMax = std::max(rect1.y + rect1.height, rect2.y + rect2.height);
    int yOverlap = (rect1.height + rect2.height) - (yMax - yMin);

    return yOverlap <= 0 ? 0 : yOverlap;
}

int computeAreaIntersection(cv::Rect const& rect1, cv::Rect const& rect2) {
    int xOverlap = computeXOverlap(rect1, rect2);
    int yOverlap = computeYOverlap(rect1, rect2);

    return xOverlap * yOverlap;
}

float computeIou(cv::Rect const& rect1, cv::Rect const& rect2) {
    int areaIntersection = computeAreaIntersection(rect1, rect2);
    return float(areaIntersection) / (rect1.area() + rect2.area() - areaIntersection);
}

int computeSpacing(cv::Rect const& rect1, cv::Rect const& rect2) {
    return std::abs(xMid(rect1) - xMid(rect2));
}

} // end namespace cuizhou