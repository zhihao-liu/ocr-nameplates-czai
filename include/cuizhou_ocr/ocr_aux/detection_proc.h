//
// Created by Zhihao Liu on 18-4-4.
//

#ifndef CUIZHOU_OCR_OCRUTILS_H
#define CUIZHOU_OCR_OCRUTILS_H

#include <vector>
#include <string>
#include "detection.h"
#include "datamodel/ocr_detection.h"


namespace cuizhou {

std::vector<std::string> readClassNames(std::string const& path, bool addBackground = false);
bool isNumbericChar(std::string const& str);

void sortByXMid(std::vector<Detection>& dets);
void sortByYMid(std::vector<Detection>& dets);
void sortByScoreDescending(std::vector<Detection>& dets);
bool isSortedByXMid(std::vector<Detection> const& dets);

OcrDetection joinDetections(std::vector<Detection> const& dets);
std::string joinDetectedChars(std::vector<Detection> const& dets);

void eliminateYOutliers(std::vector<Detection>& dets, float thresh = 0.25);
void eliminateLetters(std::vector<Detection>& dets);

cv::Rect computeExtent(std::vector<Detection> const& dets);
double estimateCharAlignmentSlope(std::vector<Detection> const& dets);
int estimateCharSpacing(std::vector<Detection> const& dets);

cv::Rect& shrinkRectToExtent(cv::Rect& rect, cv::Rect const& extentInRect);
cv::Rect& expandRect(cv::Rect& rect, int xBorder, int yBorder);
bool isRectTooLarge(cv::Rect const& rect, cv::Rect const& extentInRect, int widthThresh, int heightThresh);

} // end namespace cuizhou


#endif //CUIZHOU_OCR_OCRUTILS_H
