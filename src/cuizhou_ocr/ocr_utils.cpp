//
// Created by Zhihao Liu on 18-4-4.
//

#include "ocr_utils.hpp"
#include <dirent.h>

#include <opencv2/imgproc/imgproc.hpp>


namespace cuizhou {

cv::Mat OcrUtils::imgResizeAndFill(cv::Mat const& img,
                                   cv::Size const& newSize,
                                   PerspectiveTransform* pForwardTransform) {
    return imgResizeAndFill(img, newSize.width, newSize.height, pForwardTransform);
}

// resize the image while preserve its width-height ratio
// save the transform parameters into pForwardTransform if it is non-null
cv::Mat OcrUtils::imgResizeAndFill(cv::Mat const& img,
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

cv::Mat OcrUtils::imgRotate(cv::Mat const& img, double angleInDegree) {
    cv::Mat newImg;

    cv::Point2d pt(img.cols / 2.0, img.rows / 2.0);
    cv::Mat r = cv::getRotationMatrix2D(pt, angleInDegree, 1.0);
    cv::warpAffine(img, newImg, r, img.size());

    return newImg;
}

std::vector<std::string> OcrUtils::readClassNames(std::string const& path) {
    std::ifstream file(path);
    std::vector<std::string> classNames;
    classNames.emplace_back("__background__");

    std::string line;
    while (getline(file, line)) {
        classNames.push_back(line);
    }
    return classNames;
};

bool OcrUtils::isNumbericChar(std::string const& str) {
    return str.length() == 1 && str[0] >= '0' && str[0] <= '9';
}

int OcrUtils::xMid(cv::Rect const& rect) {
    return rect.x + int(round(rect.width / 2.0));
}

int OcrUtils::yMid(cv::Rect const& rect) {
    return rect.y + int(round(rect.height / 2.0));
}

int OcrUtils::computeXOverlap(cv::Rect const& rect1, cv::Rect const& rect2) {
    int xMin = std::min(rect1.x, rect2.x);
    int xMax = std::max(rect1.x + rect1.width, rect2.x + rect2.width);
    int xOverlap = (rect1.width + rect2.width) - (xMax - xMin);

    return xOverlap <= 0 ? 0 : xOverlap;
}

int OcrUtils::computeYOverlap(cv::Rect const& rect1, cv::Rect const& rect2) {
    int yMin = std::min(rect1.y, rect2.y);
    int yMax = std::max(rect1.y + rect1.height, rect2.y + rect2.height);
    int yOverlap = (rect1.height + rect2.height) - (yMax - yMin);

    return yOverlap <= 0 ? 0 : yOverlap;
}

int OcrUtils::computeAreaIntersection(cv::Rect const& rect1, cv::Rect const& rect2) {
    int xOverlap = computeXOverlap(rect1, rect2);
    int yOverlap = computeYOverlap(rect1, rect2);

    return xOverlap * yOverlap;
}

float OcrUtils::computeIou(cv::Rect const& rect1, cv::Rect const& rect2) {
    int areaIntersection = computeAreaIntersection(rect1, rect2);
    return float(areaIntersection) / (rect1.area() + rect2.area() - areaIntersection);
}

int OcrUtils::computeSpacing(cv::Rect const& rect1, cv::Rect const& rect2) {
    return std::abs(xMid(rect1) - xMid(rect2));
}

cv::Rect OcrUtils::validateRoi(cv::Rect const& roi, int width, int height) {
    int newX = std::min(std::max(roi.x, 0), width - 1);
    int newY = std::min(std::max(roi.y, 0), height - 1);
    int newW = std::min(std::max(roi.width, 1), width - newX);
    int newH = std::min(std::max(roi.height, 1), height - newY);

    return cv::Rect(newX, newY, newW, newH);
}

cv::Rect OcrUtils::validateRoi(cv::Rect const& roi, cv::Mat const& img) {
    // ensure the roi is within the extent of the image after adjustments
    return validateRoi(roi, img.cols, img.rows);
}

cv::Rect OcrUtils::validateRoi(cv::Rect const& roi, cv::Rect const& extent) {
    // ensure the roi is within the extent of the image after adjustments
    return validateRoi(roi, extent.width, extent.height);
}

LeastSquare::LeastSquare(std::vector<double> const& x, std::vector<double> const& y) {
    double t1 = 0, t2 = 0, t3 = 0, t4 = 0;
    for (int i = 0; i < x.size(); ++i) {
        t1 += x[i] * x[i];
        t2 += x[i];
        t3 += x[i] * y[i];
        t4 += y[i];
    }
    a = (t3 * x.size() - t2 * t4) / (t1 * x.size() - t2 * t2);
    b = (t1 * t4 - t2 * t3) / (t1 * x.size() - t2 * t2);
}

double LeastSquare::getSlope() const {
    return a;
};

double LeastSquare::getConstant() const {
    return b;
};

} // end namespace cuizhou