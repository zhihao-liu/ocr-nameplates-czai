//
// Created by Zhihao Liu on 18-4-4.
//

#include "OcrNameplatesAlfa.h"
#include "OcrUtils.hpp"


using namespace cuizhou;

int const OcrNameplatesAlfa::ROI_X_BORDER = 8;
int const OcrNameplatesAlfa::ROI_Y_BORDER = 4;
int const OcrNameplatesAlfa::CHAR_X_BORDER = 2;
int const OcrNameplatesAlfa::CHAR_Y_BORDER = 1;

OcrNameplatesAlfa::~OcrNameplatesAlfa() = default;

OcrNameplatesAlfa::OcrNameplatesAlfa() = default;

OcrNameplatesAlfa::OcrNameplatesAlfa(PvaDetector& detectorKeys, PvaDetector& detectorValues)
        : _pDetectorKeys(&detectorKeys), _pDetectorValues(&detectorValues) {}

void OcrNameplatesAlfa::processImage() {
    detectKeys();
    auto itrKeyVin = _keyDetectedItems.find(CLASSNAME_VIN);
    if (itrKeyVin != _keyDetectedItems.end()) {
        DetectedItem valueVin = detectValueVin();
        KeyValuePair keyValuePairVin = {itrKeyVin->second, valueVin};
        _result.put(CLASSNAME_VIN, keyValuePairVin);
    }
}

void OcrNameplatesAlfa::detectKeys() {
    _pDetectorKeys->setThresh(0.5, 0.1);

    std::vector<Detection> keyDets = _pDetectorKeys->detect(_image);
    for (auto const& keyDet: keyDets) {
        std::string keyName = keyDet.getClass();
        DetectedItem detectedItem = {keyName, keyDet.getRect()};
        _keyDetectedItems.emplace(keyName, detectedItem);
    }
}

DetectedItem OcrNameplatesAlfa::detectValueVin() {
    auto itrKeyVin = _keyDetectedItems.find(CLASSNAME_VIN);
    if (itrKeyVin == _keyDetectedItems.end()) return DetectedItem();

    _pDetectorValues->setThresh(0.05, 0.3);

    cv::Rect keyRect = itrKeyVin->second.rect;
    cv::Rect valueRect = estimateValueRectOfVin(keyRect);
    std::vector<Detection> valueDets = _pDetectorValues->detect(_image(valueRect));

    sortByXMid(valueDets);
    eliminateYOutliers(valueDets);
    eliminateOverlaps(valueDets);

    double slope = computeCharAlignmentSlope(valueDets);
    if (std::abs(slope) > 0.025) {
        double angle = std::atan(slope) / CV_PI * 180;
        OcrUtils::imrotate(_image, _image, angle);
        detectKeys();

        itrKeyVin = _keyDetectedItems.find(CLASSNAME_VIN);
        if (itrKeyVin == _keyDetectedItems.end()) return DetectedItem();

        keyRect = itrKeyVin->second.rect;
        valueRect = estimateValueRectOfVin(keyRect);
        valueDets = _pDetectorValues->detect(_image(valueRect));

        sortByXMid(valueDets);
        eliminateYOutliers(valueDets);
        eliminateOverlaps(valueDets);
    }

    // second round in the network
    adjustRoi(valueRect, computeExtent(valueDets));
    expandRoi(valueRect, valueDets);
    OcrUtils::validateWindow(valueRect, _image);
    valueDets = _pDetectorValues->detect(_image(valueRect));

    sortByXMid(valueDets);
    eliminateYOutliers(valueDets);
    eliminateOverlaps(valueDets);

    cv::Rect detsExtent = computeExtent(valueDets);
    if (isRoiTooLarge(valueRect, detsExtent)) {
        // third round in the network
        adjustRoi(valueRect, detsExtent);
        OcrUtils::validateWindow(valueRect, _image);
        valueDets = _pDetectorValues->detect(_image(valueRect));

        sortByXMid(valueDets);
        eliminateYOutliers(valueDets);
        eliminateOverlaps(valueDets);
    }

    if (valueDets.size() < 17) {
        // fourth round in the network
        addGapDetections(valueDets, valueRect);
        sortByXMid(valueDets);
    }

    if (valueDets.size() > 17) {
        mergeOverlappedDetections(valueDets);
    }

    std::string value = joinDetectedChars(valueDets);
    return {value, valueRect};
}

cv::Rect OcrNameplatesAlfa::estimateValueRectOfVin(cv::Rect const& keyRect) {
    return cv::Rect(keyRect.x + keyRect.width + 5,
                    int(std::round(keyRect.y - keyRect.height * 0.25)),
                    int(std::round(keyRect.width * 1.75)),
                    int(std::round(keyRect.height * 1.5)));
}

void OcrNameplatesAlfa::sortByXMid(std::vector<Detection>& dets) {
    std::sort(dets.begin(), dets.end(), [](Detection const& det1, Detection const& det2) {
        return OcrUtils::xMid(det1.getRect()) < OcrUtils::xMid(det2.getRect());
    });
}

bool OcrNameplatesAlfa::isSortedByXMid(std::vector<Detection> const& dets) {
    return std::is_sorted(dets.begin(), dets.end(), [](Detection const& det1, Detection const& det2) {
        return OcrUtils::xMid(det1.getRect()) < OcrUtils::xMid(det2.getRect());
    });
}

std::string OcrNameplatesAlfa::joinDetectedChars(std::vector<Detection> const& dets) {
    assert(isSortedByXMid(dets));

    std::string str = "";
    for (auto const& det: dets) {
        str += det.getClass();
    }
    return str;
}

bool OcrNameplatesAlfa::containsUnambiguousNumberOne(Detection const& det1, Detection const& det2) {
    return (det1.getClass() == "1" && det2.getClass() != "7" && det2.getClass() != "4" && det2.getClass() != "H")
           || (det2.getClass() == "1" && det1.getClass() != "7" && det1.getClass() != "4" && det1.getClass() != "H");
}

void OcrNameplatesAlfa::eliminateYOutliers(std::vector<Detection>& dets) {
    if (dets.size() < 3) return;

    // remove those lies off the horizontal reference line
    int heightRef = OcrUtils::findItemWithMedian(dets, [](Detection const& det1, Detection const& det2) {
        return det1.getRect().height < det2.getRect().height;
    }).getRect().height;

    int yMidRef = OcrUtils::yMid(OcrUtils::findItemWithMedian(dets, [](Detection const& det1, Detection const& det2) {
        return OcrUtils::yMid(det1.getRect()) < OcrUtils::yMid(det2.getRect());
    }).getRect());

    for (auto itr = dets.begin(); itr != dets.end();) {
        if (std::abs(OcrUtils::yMid(itr->getRect()) - yMidRef) > 0.25 * heightRef) {
            itr = dets.erase(itr);
        } else {
            ++itr;
        }
    }
}

void OcrNameplatesAlfa::eliminateOverlaps(std::vector<Detection>& dets) {
    if (dets.size() < 2) return;
    assert(isSortedByXMid(dets));

    for (auto itr = dets.begin() + 1; itr != dets.end();) {
        // set larger tolerance for "1" because it is overlapped most of the time
        double firstTol = containsUnambiguousNumberOne(*(itr - 1), *itr) ? 0.6 : 0.4;
        double secondTol = containsUnambiguousNumberOne(*(itr - 1), *itr) ? 0.6 : 0.3;

        double overlap = OcrUtils::computeIou((itr - 1)->getRect(), itr->getRect());
        if (overlap > firstTol) {
            if ((itr - 1)->getScore() < itr->getScore()) {
                itr = dets.erase(itr - 1) + 1;
                continue;
            } else {
                itr = dets.erase(itr);
                continue;
            }
        } else if (overlap > secondTol) {
            if ((itr - 1)->getScore() < itr->getScore() && (itr - 1)->getScore() < 0.2) {
                itr = dets.erase(itr - 1) + 1;
                continue;
            } else if (itr->getScore() < (itr - 1)->getScore() && itr->getScore() < 0.2) {
                itr = dets.erase(itr);
                continue;
            }
        };

        ++itr;
    }
}

cv::Rect OcrNameplatesAlfa::computeExtent(std::vector<Detection> const& dets) {
    if (dets.empty()) return cv::Rect();

    int left = INT_MAX, right = INT_MIN, top = INT_MAX, bottom = INT_MIN;
    for (auto const& det: dets) {
        if (det.getRect().x < left) left = det.getRect().x;
        if (det.getRect().x + det.getRect().width > right) right = det.getRect().x + det.getRect().width;
        if (det.getRect().y < top) top = det.getRect().y;
        if (det.getRect().y + det.getRect().height > bottom) bottom = det.getRect().y + det.getRect().height;
    }

    return cv::Rect(left, top, right - left, bottom - top);
}

double OcrNameplatesAlfa::computeCharAlignmentSlope(std::vector<Detection> const& dets) {
    if (dets.empty()) return 0;

    std::vector<double> xCoords, yCoords;
    std::transform(dets.begin(), dets.end(), back_inserter(xCoords),
                   [](Detection det) { return OcrUtils::xMid(det.getRect()); });
    std::transform(dets.begin(), dets.end(), back_inserter(yCoords),
                   [](Detection det) { return OcrUtils::yMid(det.getRect()); });

    LeastSquare ls(xCoords, yCoords);
    return ls.getSlope();
}

cv::Rect& OcrNameplatesAlfa::expandRoi(cv::Rect& roi, std::vector<Detection> const& dets) {
    assert(isSortedByXMid(dets));

    int vacancy = 17 - int(dets.size());
    if (vacancy <= 0) return roi;

    double charWidth = double(computeExtent(dets).width) / dets.size();
    double additionalWidth = charWidth * vacancy * 1.05;

    int newX = roi.x;
    int newY = roi.y;
    int newW = int(round(roi.width + additionalWidth));
    int newH = roi.height;

    double slope = computeCharAlignmentSlope(dets);

    if (slope > 0) {
        newH += round(slope * additionalWidth);
    } else {
        newY += round(slope * additionalWidth);
    }

    roi.x = newX;
    roi.y = newY;
    roi.width = newW;
    roi.height = newH;

    return roi;
}

bool OcrNameplatesAlfa::isRoiTooLarge(cv::Rect const& roi, cv::Rect const& detsExtent) {
    return (roi.width - detsExtent.width > 2.5 * ROI_X_BORDER) || (roi.height - detsExtent.height > 2.5 * ROI_Y_BORDER);
}

cv::Rect& OcrNameplatesAlfa::adjustRoi(cv::Rect& roi, cv::Rect const& detsExtent) {
    int newLeft = roi.x + (detsExtent.x - ROI_X_BORDER);
    int newRight = roi.x + (detsExtent.x + detsExtent.width + ROI_X_BORDER);
    int newTop = roi.y + (detsExtent.y - ROI_Y_BORDER);
    int newBottom = roi.y + (detsExtent.y + detsExtent.height + ROI_Y_BORDER);

    roi.x = newLeft;
    roi.y = newTop;
    roi.width = newRight - newLeft;
    roi.height = newBottom - newTop;

    return roi;
}

int OcrNameplatesAlfa::estimateCharSpacing(std::vector<Detection> const& dets) {
    if (dets.size() < 2) return 0;
    assert(&& isSortedByXMid(dets));

    std::vector<int> spacings;
    for (auto itr = dets.begin() + 1; itr != dets.end(); ++itr) {
        int spacing = OcrUtils::computeSpacing((itr - 1)->getRect(), itr->getRect());
        spacings.push_back(spacing);
    }

    return OcrUtils::findItemWithMedian(spacings, std::less<int>());
}

void OcrNameplatesAlfa::addGapDetections(std::vector<Detection>& dets, cv::Rect const& roi) const {
    if (dets.size() <= 2 || dets.size() >= 17) return;
    assert(isSortedByXMid(dets));

    int spacingRef = estimateCharSpacing(dets);
    for (auto itr = dets.begin() + 1; itr != dets.end(); ++itr) {
        cv::Rect leftRect = (itr - 1)->getRect();
        cv::Rect rightRect = itr->getRect();
        if (OcrUtils::computeSpacing(leftRect, rightRect) > 1.5 * spacingRef) {
            int gapX = leftRect.x + leftRect.width - CHAR_X_BORDER;
            int gapY = (leftRect.y + rightRect.y) / 2 - CHAR_Y_BORDER;
            int gapW = (rightRect.x - (leftRect.x + leftRect.width)) + CHAR_X_BORDER * 2;
            int gapH = (leftRect.height + rightRect.height) / 2 + CHAR_Y_BORDER * 2;

            int gapXReal = roi.x + gapX;
            int gapYReal = roi.y + gapY;

            cv::Rect gapRect;
            gapRect.x = gapXReal;
            gapRect.y = gapYReal;
            gapRect.width = gapW;
            gapRect.height = gapH;

            OcrUtils::validateWindow(gapRect, _image);
            cv::Mat gapExpanded(gapRect.height, gapRect.width * 10, CV_8UC3, cv::Scalar(0, 0, 0));
            cv::Rect expandedCenter(cv::Point(gapExpanded.cols / 2 - gapRect.width/2, 0), gapRect.size());

            _image(gapRect).copyTo(gapExpanded(expandedCenter));
            std::vector<Detection> gapDets = _pDetectorValues->detect(gapExpanded);

            if (!gapDets.empty()) {
                gapRect.x = gapX;
                gapRect.y = gapY;
                OcrUtils::validateWindow(gapRect, roi);

                Detection& gapDet = gapDets.front();
                gapDet.setRect(gapRect);

                dets.push_back(gapDet);
            }
        }
    }
}

void OcrNameplatesAlfa::mergeOverlappedDetections(std::vector<Detection>& dets) {
    if (dets.size() <= 17) return;

    std::vector<double> overlaps;
    for (auto itr = dets.begin() + 1; itr != dets.end(); ++itr) {
        double iou = OcrUtils::computeIou((itr - 1)->getRect(), itr->getRect());
        overlaps.push_back(iou);
    }

    double maxOverlap = overlaps[0];
    int maxIdx = 0;
    for (int i = 0; i < overlaps.size(); ++i) {
        if (overlaps[i] > maxOverlap) {
            maxOverlap = overlaps[i];
            maxIdx = i;
        }
    }

    int idxToErase = dets[maxIdx].getScore() > dets[maxIdx + 1].getScore() ? maxIdx + 1 : maxIdx;
    dets.erase(dets.begin() + idxToErase);

    mergeOverlappedDetections(dets);
}