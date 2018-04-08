//
// Created by Zhihao Liu on 18-4-4.
//

#include "OcrUtils.hpp"
#include <dirent.h>

#include <opencv2/imgproc/imgproc.hpp>

using namespace cuizhou;

void OcrUtils::imrotate(cv::Mat& img, cv::Mat& newImg, double angleInDegree) {
    cv::Point2d pt(img.cols / 2.0, img.rows / 2.0);
    cv::Mat r = cv::getRotationMatrix2D(pt, angleInDegree, 1.0);
    cv::warpAffine(img, newImg, r, img.size());
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

double OcrUtils::computeIou(cv::Rect const& rect1, cv::Rect const& rect2) {
    int areaIntersection = computeAreaIntersection(rect1, rect2);
    return double(areaIntersection) / (rect1.area() + rect2.area() - areaIntersection);
}

int OcrUtils::computeSpacing(cv::Rect const& rect1, cv::Rect const& rect2) {
    return std::abs(xMid(rect1) - xMid(rect2));
}

cv::Rect& OcrUtils::validateWindow(cv::Rect& window, int width, int height) {
    if (window.x < 0) {
        window.x = 0;
    } else if (window.x >= width) {
        window.x = width - 1;
    }

    if (window.y < 0) {
        window.y = 0;
    } else if (window.y >= height) {
        window.y = height - 1;
    }

    if (window.width < 1) {
        window.width = 1;
    } else if (window.x + window.width > width) {
        window.width = width - window.x;
    }

    if (window.height < 1) {
        window.height = 1;
    } else if (window.y + window.height > height) {
        window.height = height - window.y;
    }

    return window;
}

cv::Rect& OcrUtils::validateWindow(cv::Rect& roi, cv::Mat const& img) {
    // ensure the roi is within the extent of the image after adjustments
    return validateWindow(roi, img.cols, img.rows);
}

cv::Rect& OcrUtils::validateWindow(cv::Rect& roi, cv::Rect const& extent) {
    // ensure the roi is within the extent of the image after adjustments
    return validateWindow(roi, extent.width, extent.height);
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