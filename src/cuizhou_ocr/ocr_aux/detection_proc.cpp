//
// Created by Zhihao Liu on 18-4-4.
//

#include "ocr_aux/detection_proc.h"
#include <fstream>
#include <limits>
#include "data_utils/cv_extension.h"
#include "data_utils/data_proc.hpp"


namespace cz {

bool isNumbericChar(std::string const& str) {
    return str.length() == 1 && str[0] >= '0' && str[0] <= '9';
}

std::vector<std::string> readClassNames(std::string const& path, bool addBackground) {
    std::ifstream file(path);
    std::vector<std::string> classNames;
    if (addBackground) classNames.emplace_back("__background__");

    std::string line;
    while (getline(file, line)) {
        classNames.push_back(std::move(line));
    }
    return classNames;
};


void sortByXMid(std::vector<Detection>& dets) {
    std::sort(dets.begin(), dets.end(),
              [](Detection const& lhs, Detection const& rhs) { return xMid(lhs.rect) < xMid(rhs.rect); });
}

void sortByYMid(std::vector<Detection>& dets) {
    std::sort(dets.begin(), dets.end(),
              [](Detection const& lhs, Detection const& rhs) { return yMid(lhs.rect) < yMid(rhs.rect); });
}

void sortByScoreDescending(std::vector<Detection>& dets) {
    std::sort(dets.begin(), dets.end(),
              [](Detection const& lhs, Detection const& rhs) { return lhs.score > rhs.score; });
}

bool isSortedByXMid(std::vector<Detection> const& dets) {
    return std::is_sorted(dets.cbegin(), dets.cend(), [](Detection const& lhs, Detection const& rhs) {
        return xMid(lhs.rect) < xMid(rhs.rect);
    });
}

OcrDetection joinDetections(std::vector<Detection> const& dets) {
    std::string text = joinDetectedChars(dets);
    cv::Rect rect = computeExtent(dets);
    return OcrDetection(text, rect);
}

std::string joinDetectedChars(std::vector<Detection> const& dets) {
    assert(isSortedByXMid(dets));

    std::string result;
    for (auto const& det : dets) {
        result.append(det.label);
    }
    return result;
}

void eliminateLetters(std::vector<Detection>& dets) {
    dets.erase(std::remove_if(dets.begin(), dets.end(),
                              [](Detection const& det) { return !isNumbericChar(det.label); }),
               dets.end());
}

void eliminateYOutliers(std::vector<Detection>& dets, float thresh) {
    if (dets.size() < 3) return;

    // remove those lies off the horizontal reference line
    int heightRef = findMedian(dets, [](Detection const& det) { return det.rect.height; });
    int yMidRef = findMedian(dets, [](Detection const& det) { return yMid(det.rect); });

    dets.erase(std::remove_if(dets.begin(), dets.end(),
                              [&](Detection const& det) {
                                  return std::abs(yMid(det.rect) - yMidRef) > thresh * heightRef;
                              }),
               dets.end());
}

cv::Rect computeExtent(std::vector<Detection> const& dets) {
    if (dets.empty()) return cv::Rect();

    cv::Rect extent = dets.front().rect;
    for (auto const& det : dets) {
        extent |= det.rect;
    }
    return extent;
}

double estimateCharAlignmentSlope(std::vector<Detection> const& dets) {
    if (dets.empty()) return 0;

    std::vector<double> xCoords, yCoords;
    std::transform(dets.cbegin(), dets.cend(), std::back_inserter(xCoords),
                   [](Detection det) { return xMid(det.rect); });
    std::transform(dets.cbegin(), dets.cend(), std::back_inserter(yCoords),
                   [](Detection det) { return yMid(det.rect); });

    LinearFit lf(xCoords, yCoords);
    return lf.slope();
}

int estimateCharSpacing(std::vector<Detection> const& dets) {
    if (dets.size() < 2) return 0;
    assert(isSortedByXMid(dets));

    std::vector<int> spacings;
    for (auto itr = std::next(dets.cbegin()); itr != dets.cend(); ++itr) {
        int spacing = computeSpacing(std::prev(itr)->rect, itr->rect);
        spacings.push_back(spacing);
    }

    return findMedian(spacings);
}

void shrinkRectToExtent(cv::Rect& rect, cv::Rect const& extentInRect) {
//    rect.x += extentInRect.x;
//    rect.y += extentInRect.y;
    rect += extentInRect.tl();
    rect.width = extentInRect.width;
    rect.height = extentInRect.height;
}

void expandRect(cv::Rect& rect, int xBorder, int yBorder) {
//    rect.x -= xBorder;
//    rect.y -= yBorder;
//    rect.width += 2 * xBorder;
//    rect.height += 2 * yBorder;
    rect -= cv::Point(xBorder, yBorder);
    rect += cv::Size(xBorder, yBorder) * 2;
}

bool isRectTooLarge(cv::Rect const& rect, cv::Rect const& extentInRect, int widthThresh, int heightThresh) {
    return (rect.width - extentInRect.width > widthThresh) ||
           (rect.height - extentInRect.height > heightThresh);
}

} // end namespace cz