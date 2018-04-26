//
// Created by Zhihao Liu on 18-4-4.
//

#include "ocr_nameplates_alfa.h"
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include "ocr_utils.hpp"


// DEBUG
cv::Mat d_imgToShow;

namespace cuizhou {

int const OcrNameplatesAlfa::STANDARD_IMG_WIDTH = 1024;
int const OcrNameplatesAlfa::STANDARD_IMG_HEIGHT = 768;
int const OcrNameplatesAlfa::ROI_X_BORDER = 8;
int const OcrNameplatesAlfa::ROI_Y_BORDER = 4;
int const OcrNameplatesAlfa::CHAR_X_BORDER = 2;
int const OcrNameplatesAlfa::CHAR_Y_BORDER = 1;

std::vector<std::string> const OcrNameplatesAlfa::PAINT_CANDIDATES = {"414", "361", "217", "248", "092", "093", "035",
                                                                      "318", "408", "409", "620"};
EnumHashMap<OcrNameplatesAlfa::NameplateField, int> const OcrNameplatesAlfa::VALUE_LENGTH = {
        {NameplateField::VIN,                     17},
        {NameplateField::MAX_MASS_ALLOWED,        4},
        {NameplateField::MAX_NET_POWER_OF_ENGINE, 3},
        {NameplateField::ENGINE_MODEL,            8},
        {NameplateField::NUM_PASSENGERS,          5},
        {NameplateField::VEHICLE_MODEL,           8},
        {NameplateField::ENGINE_DISPLACEMENT,     4},
        {NameplateField::DATE_OF_MANUFACTURE,     6},
        {NameplateField::PAINT,                   3}
};

OcrNameplatesAlfa::OcrNameplatesAlfa(Detector detectorKeys,
                                     Detector detectorValuesVin,
                                     Detector detectorValuesStitched,
                                     Classifier classifierChars)
        : detectorKeys_(std::move(detectorKeys)),
          detectorValuesVin_(std::move(detectorValuesVin)),
          detectorValuesStitched_(std::move(detectorValuesStitched)),
          classifierChars_(std::move(classifierChars)) {}

void OcrNameplatesAlfa::processImage() {
    result_.clear();

    image_ = OcrUtils::imgResizeAndFill(image_, STANDARD_IMG_WIDTH, STANDARD_IMG_HEIGHT);

    detectKeys();
    adaptiveRotationWithUpdatingKeyDetections();

    detectValueOfVin();
    detectValuesOfOtherCodeFields();
}

void OcrNameplatesAlfa::detectKeys() {
    keyOcrDetections_.clear();

    detectorKeys_.setThresh(0.5, 0.1); // fixed params, empirical

    std::vector<Detection> keyDets = detectorKeys_.detect(image_);
    std::transform(keyDets.cbegin(), keyDets.cend(),
                   std::inserter(keyOcrDetections_, keyOcrDetections_.end()),
                   [](Detection const& det) {
                       std::string const& keyName = det.label;
                       OcrDetection keyItem(keyName, det.rect);
                       return std::make_pair(fieldDict_.toEnum(keyName), keyItem);
                   });
}

void OcrNameplatesAlfa::adaptiveRotationWithUpdatingKeyDetections() {
    auto itrKeyVin = keyOcrDetections_.find(NameplateField::VIN);
    if (itrKeyVin == keyOcrDetections_.end()) return;

    detectorValuesVin_.setThresh(0.1, 0.3);

    cv::Rect const& keyRoi = itrKeyVin->second.rect;
    cv::Rect valueRoi = estimateValueRoi(NameplateField::VIN, keyRoi);
    std::vector<Detection> valueDets = detectorValuesVin_.detect(image_(valueRoi));

    sortByXMid(valueDets);
    eliminateXOverlaps(valueDets, NameplateField::VIN);
    eliminateYOutliers(valueDets);

    double slope = estimateCharAlignmentSlope(valueDets);
    if (std::abs(slope) > 0.025) {
        double angle = std::atan(slope) / CV_PI * 180;
        image_ = OcrUtils::imgRotate(image_, angle);
        detectKeys(); // update detections of keys
    }
}

void OcrNameplatesAlfa::detectValueOfVin() {
    auto itrKeyItemVin = keyOcrDetections_.find(NameplateField::VIN);
    if (itrKeyItemVin == keyOcrDetections_.end()) return;
    OcrDetection const& keyItem = itrKeyItemVin->second;

    detectorValuesVin_.setThresh(0.05, 0.3);

    cv::Rect keyRoi = keyItem.rect;
    cv::Rect valueRoi = estimateValueRoi(NameplateField::VIN, keyRoi);
    OcrUtils::validateRoi(valueRoi, image_);
    // no need to resize and fill because the model for VIN is trained with stretched images
    std::vector<Detection> valueDets = detectorValuesVin_.detect(image_(valueRoi));

    sortByXMid(valueDets);
    eliminateXOverlaps(valueDets, NameplateField::VIN);
    eliminateYOutliers(valueDets);

    // second round in the network
    adjustRoi(valueRoi, computeExtent(valueDets));
    expandRoi(valueRoi, valueDets);
    OcrUtils::validateRoi(valueRoi, image_);
    valueDets = detectorValuesVin_.detect(image_(valueRoi));

    sortByXMid(valueDets);
    eliminateXOverlaps(valueDets, NameplateField::VIN);
    eliminateYOutliers(valueDets);

    cv::Rect detsExtent = computeExtent(valueDets);
    if (isRoiTooLarge(valueRoi, detsExtent)) {
        // third round in the network
        adjustRoi(valueRoi, detsExtent);
        OcrUtils::validateRoi(valueRoi, image_);
        valueDets = detectorValuesVin_.detect(image_(valueRoi));

        sortByXMid(valueDets);
        eliminateXOverlaps(valueDets, NameplateField::VIN);
        eliminateYOutliers(valueDets);
    }

    if (valueDets.size() < 17) {
        // fourth round in the network
        addGapDetections(valueDets, valueRoi);
        sortByXMid(valueDets);
    }

    if (valueDets.size() > 17) {
        mergeOverlappedDetections(valueDets);
    }

    updateByClassification(valueDets, image_(valueRoi));

    std::string valueText = joinDetectedChars(valueDets);
    OcrDetection valueItem(valueText, valueRoi);
    result_.emplace(NameplateField::VIN, KeyValueDetection(keyItem, valueItem));
}

cv::Rect OcrNameplatesAlfa::estimateValueRoi(NameplateField field, cv::Rect const& keyRoi) {
    switch (field) {
        case NameplateField::VIN: {
            return cv::Rect(keyRoi.br().x + 5,
                            keyRoi.y - int(std::round(keyRoi.height * 0.25)),
                            int(std::round(keyRoi.width * 1.75)),
                            int(std::round(keyRoi.height * 1.5)));
        }

        case NameplateField::MAX_MASS_ALLOWED: {
            return cv::Rect(keyRoi.br().x,
                            keyRoi.y - int(std::round(keyRoi.height * -0.1)),
                            std::max(int(keyRoi.width * 0.7), 100),
                            std::min(int(keyRoi.height * 1.25), 50));
        }

        case NameplateField::MAX_NET_POWER_OF_ENGINE: {
            return cv::Rect(keyRoi.br().x + 5,
                            keyRoi.y - int(std::round(keyRoi.height * -0.1)),
                            std::max(int(keyRoi.width * 0.5), 95),
                            std::min(int(keyRoi.height * 1.25), 55));
        }

        case NameplateField::ENGINE_MODEL: {
            return cv::Rect(keyRoi.br().x + 10,
                            keyRoi.y - int(std::round(keyRoi.height * 0.1)),
                            std::max(int(keyRoi.width * 1.0), 150),
                            std::min(int(keyRoi.height * 1.25), 40));
        }

        case NameplateField::NUM_PASSENGERS: {
            return cv::Rect(keyRoi.br().x + 5,
                            keyRoi.y - int(std::round(keyRoi.height * 0.05)),
                            std::max(int(keyRoi.width * 0.4), 40),
                            std::min(int(keyRoi.height * 1.15), 40));
        }

        case NameplateField::VEHICLE_MODEL: {
            return cv::Rect(keyRoi.br().x + 5,
                            keyRoi.y - int(std::round(keyRoi.height * 0.1)),
                            std::max(int(keyRoi.width * 1.2), 120),
                            std::min(int(keyRoi.height * 1.25), 40));
        }

        case NameplateField::ENGINE_DISPLACEMENT: {
            return cv::Rect(keyRoi.br().x + 10,
                            keyRoi.y - int(std::round(keyRoi.height * 0.1)),
                            std::max(int(keyRoi.width * 0.8), 80),
                            std::min(int(keyRoi.height * 1.25), 40));
        }

        case NameplateField::DATE_OF_MANUFACTURE: {
            return cv::Rect(keyRoi.br().x + 10,
                            keyRoi.y - int(std::round(keyRoi.height * 0.1)),
                            std::max(int(keyRoi.width * 1.0), 75),
                            std::min(int(keyRoi.height * 1.25), 40));
        }

        case NameplateField::PAINT: {
            return cv::Rect(keyRoi.br().x + 5,
                            keyRoi.y - int(std::round(keyRoi.height * 0.1)),
                            std::max(int(keyRoi.width * 1.2), 45),
                            std::min(int(keyRoi.height * 1.25), 35));
        }

        default: {
            return cv::Rect();
        }
    }
}

void OcrNameplatesAlfa::sortByXMid(std::vector<Detection>& dets) {
    std::sort(dets.begin(), dets.end(), [](Detection const& lhs, Detection const& rhs) {
        return OcrUtils::xMid(lhs.rect) < OcrUtils::xMid(rhs.rect);
    });
}

void OcrNameplatesAlfa::sortByYMid(std::vector<Detection>& dets) {
    std::sort(dets.begin(), dets.end(), [](Detection const& lhs, Detection const& rhs) {
        return OcrUtils::yMid(lhs.rect) < OcrUtils::yMid(rhs.rect);
    });
}

void OcrNameplatesAlfa::sortByScoreDescending(std::vector<Detection>& dets) {
    std::sort(dets.begin(), dets.end(),
              [](Detection const& lhs, Detection const& rhs) { return lhs.score > rhs.score; });
}

bool OcrNameplatesAlfa::isSortedByXMid(std::vector<Detection> const& dets) {
    return std::is_sorted(dets.cbegin(), dets.cend(),
                          [](Detection const& lhs, Detection const& rhs) {
                              return OcrUtils::xMid(lhs.rect) < OcrUtils::xMid(rhs.rect);
                          });
}

std::string OcrNameplatesAlfa::joinDetectedChars(std::vector<Detection> const& dets) {
    assert(isSortedByXMid(dets));

    return std::accumulate(dets.cbegin(), dets.cend(), std::string(),
                           [](std::string const& res, Detection const& det) { return res + det.label; });
}

bool OcrNameplatesAlfa::containsUnambiguousNumberOne(Detection const& lhs, Detection const& rhs) {
    return (lhs.label == "1" && rhs.label != "7" && rhs.label != "4" && rhs.label != "H")
           || (rhs.label == "1" && lhs.label != "7" && lhs.label != "4" && lhs.label != "H");
}

void OcrNameplatesAlfa::eliminateYOutliers(std::vector<Detection>& dets) {
    if (dets.size() < 3) return;

    // remove those lies off the horizontal reference line
    int heightRef = OcrUtils::findMedian(dets, [](Detection const& det) { return det.rect.height; });

    int yMidRef = OcrUtils::findMedian(dets, [](Detection const& det) { return OcrUtils::yMid(det.rect); });

    for (auto itr = dets.begin(); itr != dets.end();) {
        if (std::abs(OcrUtils::yMid(itr->rect) - yMidRef) > 0.25 * heightRef) {
            itr = dets.erase(itr);
        } else {
            ++itr;
        }
    }
}

void OcrNameplatesAlfa::eliminateXOverlaps(std::vector<Detection>& dets, NameplateField field){
    std::pair<float, float> firstThresh, secondThresh;
    switch (field) {
        // for fields of which value characters are compact
        case NameplateField::VIN:
        case NameplateField::PAINT: {
            firstThresh = std::make_pair(0.6, 0.4);
            secondThresh = std::make_pair(0.6, 0.3);
        } break;

        default: {
            firstThresh = std::make_pair(0.4, 0.3);
            secondThresh = std::make_pair(0.4, 0.2);
        } break;
    }
}

void OcrNameplatesAlfa::eliminateXOverlaps(std::vector<Detection>& dets, std::pair<float, float> firstThresh,
                                           std::pair<float, float> secondThresh) {
    if (dets.size() < 2) return;
    assert(isSortedByXMid(dets));

    for (auto itr = std::next(dets.begin()); itr != dets.end();) {
        // set larger tolerance for "1" because it is overlapped most of the time
        float firstTol = containsUnambiguousNumberOne(*std::prev(itr), *itr) ? firstThresh.first : firstThresh.second;
        float secondTol = containsUnambiguousNumberOne(*std::prev(itr), *itr) ? secondThresh.first
                                                                              : secondThresh.second;

        float overlap = OcrUtils::computeIou(std::prev(itr)->rect, itr->rect);
        if (overlap > firstTol) {
            if (std::prev(itr)->score < itr->score) {
                itr = std::next(dets.erase(std::prev(itr)));
                continue;
            } else {
                itr = dets.erase(itr);
                continue;
            }
        } else if (overlap > secondTol) {
            if (std::prev(itr)->score < itr->score && std::prev(itr)->score < 0.2) {
                itr = std::next(dets.erase(std::prev(itr)));
                continue;
            } else if (itr->score < std::prev(itr)->score && itr->score < 0.2) {
                itr = dets.erase(itr);
                continue;
            }
        };

        ++itr;
    }
}

cv::Rect OcrNameplatesAlfa::computeExtent(std::vector<Detection> const& dets) {
    if (dets.empty()) return cv::Rect();

    int left = std::numeric_limits<int>::max();
    int right = std::numeric_limits<int>::min();
    int top = std::numeric_limits<int>::max();
    int bottom = std::numeric_limits<int>::min();

    for (auto const& det : dets) {
        left = std::min(left, det.rect.x);
        right = std::max(right, det.rect.br().x);
        top = std::min(top, det.rect.y);
        bottom = std::max(bottom, det.rect.br().y);
    }

    return cv::Rect(left, top, right - left + 1, bottom - top + 1);
}

double OcrNameplatesAlfa::estimateCharAlignmentSlope(std::vector<Detection> const& dets) {
    if (dets.empty()) return 0;

    std::vector<double> xCoords, yCoords;
    std::transform(dets.cbegin(), dets.cend(), std::back_inserter(xCoords),
                   [](Detection det) { return OcrUtils::xMid(det.rect); });
    std::transform(dets.cbegin(), dets.cend(), std::back_inserter(yCoords),
                   [](Detection det) { return OcrUtils::yMid(det.rect); });

    LeastSquare ls(xCoords, yCoords);
    return ls.getSlope();
}

cv::Rect& OcrNameplatesAlfa::expandRoi(cv::Rect& roi, std::vector<Detection> const& dets) {
    assert(isSortedByXMid(dets));

    int vacancy = 17 - int(dets.size());
    if (vacancy <= 0) return roi;

    float charWidth = float(computeExtent(dets).width) / dets.size();
    float additionalWidth = float(charWidth * vacancy * 1.05);

    roi.width = int(std::round(roi.width + additionalWidth));

    double slope = estimateCharAlignmentSlope(dets);
    if (slope > 0) {
        roi.height += std::round(slope * additionalWidth);
    } else {
        roi.y += std::round(slope * additionalWidth);
    }

    return roi;
}

bool OcrNameplatesAlfa::isRoiTooLarge(cv::Rect const& roi, cv::Rect const& detsExtent) {
    return (roi.width - detsExtent.width > 2.5 * ROI_X_BORDER) ||
           (roi.height - detsExtent.height > 2.5 * ROI_Y_BORDER);
}

cv::Rect& OcrNameplatesAlfa::adjustRoi(cv::Rect& roi, cv::Rect const& detsExtentInRoi) {
    roi.x += (detsExtentInRoi.x - ROI_X_BORDER);
    roi.y += (detsExtentInRoi.y - ROI_Y_BORDER);
    roi.width = detsExtentInRoi.width + ROI_X_BORDER * 2;
    roi.height = detsExtentInRoi.height + ROI_Y_BORDER * 2;
    return roi;
}

int OcrNameplatesAlfa::estimateCharSpacing(std::vector<Detection> const& dets) {
    if (dets.size() < 2) return 0;
    assert(isSortedByXMid(dets));

    std::vector<int> spacings;
    for (auto itr = std::next(dets.cbegin()); itr != dets.cend(); ++itr) {
        int spacing = OcrUtils::computeSpacing(std::prev(itr)->rect, itr->rect);
        spacings.push_back(spacing);
    }

    return OcrUtils::findMedian(spacings, [](int spacing) { return spacing; });
}

void OcrNameplatesAlfa::addGapDetections(std::vector<Detection>& dets, cv::Rect const& roi) {
    if (dets.size() <= 2 || dets.size() >= 17) return;
    assert(isSortedByXMid(dets));

    detectorValuesVin_.setThresh(0.05, 0.3);
    std::vector<Detection> addedDets;

    int spacingRef = estimateCharSpacing(dets);
    for (auto itr = std::next(dets.cbegin()); itr != dets.cend(); ++itr) {
        cv::Rect leftRect = std::prev(itr)->rect;
        cv::Rect rightRect = itr->rect;
        if (OcrUtils::computeSpacing(leftRect, rightRect) > 1.5 * spacingRef) {
            int gapX = leftRect.x + leftRect.width - CHAR_X_BORDER;
            int gapY = (leftRect.y + rightRect.y) / 2 - CHAR_Y_BORDER;
            int gapW = (rightRect.x - (leftRect.x + leftRect.width)) + CHAR_X_BORDER * 2;
            int gapH = (leftRect.height + rightRect.height) / 2 + CHAR_Y_BORDER * 2;

            int gapXReal = roi.x + gapX;
            int gapYReal = roi.y + gapY;

            cv::Rect gapRect(gapXReal, gapYReal, gapW, gapH);
            OcrUtils::validateRoi(gapRect, image_);

            cv::Mat gapExpanded(gapRect.height, gapRect.width * 10, CV_8UC3, cv::Scalar(0, 0, 0));
            cv::Rect centerRegion(cv::Point(gapExpanded.cols / 2 - gapRect.width / 2, 0), gapRect.size());

            image_(gapRect).copyTo(gapExpanded(centerRegion));
            std::vector<Detection> gapDets = detectorValuesVin_.detect(gapExpanded);

            if (!gapDets.empty()) {
                gapRect.x = gapX;
                gapRect.y = gapY;
                OcrUtils::validateRoi(gapRect, roi);

                Detection& gapDet = gapDets.front();
                gapDet.rect = gapRect;

                addedDets.push_back(std::move(gapDet));
            }
        }
    }

    dets.insert(dets.end(), addedDets.cbegin(), addedDets.cend());
}

void OcrNameplatesAlfa::updateByClassification(std::vector<Detection>& dets, cv::Mat const& srcImg) const {
    for (auto& det : dets) {
        if (!OcrUtils::isNumbericChar(det.label)) continue;
        if (det.score >= 0.8) continue;

        std::vector<Classification> clss = classifierChars_.classify(srcImg(det.rect), 1);
        if (clss.front().score > 0.9) {
            det.label = clss.front().label;
            det.score = clss.front().score;
        }
    }
};

void OcrNameplatesAlfa::mergeOverlappedDetections(std::vector<Detection>& dets) {
    if (dets.size() <= 17) return;

    std::vector<float> overlaps;
    for (auto itr = std::next(dets.cbegin()); itr != dets.cend(); ++itr) {
        float iou = OcrUtils::computeIou(std::prev(itr)->rect, itr->rect);
        overlaps.push_back(iou);
    }

    auto itrMaxOverlap = std::max_element(overlaps.cbegin(), overlaps.cend());
    int idxMaxOverlap = int(std::distance(overlaps.cbegin(), itrMaxOverlap));

    int idxToErase = dets[idxMaxOverlap].score < dets[idxMaxOverlap + 1].score ? idxMaxOverlap : idxMaxOverlap + 1;
    dets.erase(std::next(dets.begin(), idxToErase));

    mergeOverlappedDetections(dets);
}

void OcrNameplatesAlfa::eliminateLetters(std::vector<Detection>& dets) {
    for (auto itr = dets.begin(); itr != dets.end(); ) {
        if (!OcrUtils::isNumbericChar(itr->label)) {
            itr = dets.erase(itr);
        } else {
            ++itr;
        }
    }
}

//std::string OcrNameplatesAlfa::matchPaintWithLengthFixed(std::string const& str) {
//    assert(str.length() == 3);
//    for (auto const& cand : PAINT_CANDIDATES) {
//        bool matched1 = (str[0] == '$' || str[0] == cand[0]);
//        bool matched2 = (str[1] == '$' || str[1] == cand[1]);
//        bool matched3 = (str[2] == '$' || str[2] == cand[2]);
//        if (matched1 && matched2 && matched3) return cand;
//    }
//    return std::string();
//}
//
//std::string OcrNameplatesAlfa::matchPaint(std::string const& str) {
//    std::string result;
//
//    if (str.length() > 3) {
//        return matchPaint(str.substr(0, 3));
//    } else if (str.length() == 3) {
//        if (result.empty()) result = matchPaintWithLengthFixed(str);
//        if (result.empty()) result = matchPaintWithLengthFixed(str.substr(0, 2) + "$");
//        if (result.empty()) result = matchPaintWithLengthFixed(str.substr(0, 1) + "$" + str.substr(2, 1));
//        if (result.empty()) result = matchPaintWithLengthFixed("$" + str.substr(1, 2));
//    } else if (str.length() == 2) {
//        if (result.empty()) result = matchPaintWithLengthFixed(str + "$");
//        if (result.empty()) result = matchPaintWithLengthFixed(str.substr(0, 1) + "$" + str.substr(1, 1));
//        if (result.empty()) result = matchPaintWithLengthFixed("$" + str);
//    }
//
//    if (result.empty()) result = str;
//    return result;
//}

void OcrNameplatesAlfa::detectValuesOfOtherCodeFields() {
    detectorValuesStitched_.setThresh(0.05, 0.3);

    std::vector<NameplateField> fields = {NameplateField::ENGINE_MODEL, NameplateField::VEHICLE_MODEL,
                                          NameplateField::MAX_MASS_ALLOWED, NameplateField::MAX_NET_POWER_OF_ENGINE,
                                          NameplateField::ENGINE_DISPLACEMENT, NameplateField::DATE_OF_MANUFACTURE,
                                          NameplateField::NUM_PASSENGERS, NameplateField::PAINT};

    std::vector<cv::Rect> targetRois = {cv::Rect(0, 0, 396, 320), cv::Rect(0, 320, 396, 320),
                                        cv::Rect(396, 0, 264, 320), cv::Rect(396, 320, 264, 320),
                                        cv::Rect(660, 0, 264, 320), cv::Rect(660, 320, 264, 320),
                                        cv::Rect(924, 0, 132, 320), cv::Rect(924, 320, 132, 320)};

    // the mapping from ROIs on the source image to the collage image
    // the first and second item of the pair<Rect, Rect> are origin ROI and target ROI respectively
    EnumHashMap<NameplateField, std::pair<cv::Rect, cv::Rect>> roiMapping;
    for (int i = 0; i < fields.size(); ++i) {
        auto itrKeyItem = keyOcrDetections_.find(fields[i]);
        if (itrKeyItem == keyOcrDetections_.end()) continue;

        cv::Rect originRoi = estimateValueRoi(fields[i], itrKeyItem->second.rect);
        OcrUtils::validateRoi(originRoi, image_);
        roiMapping.emplace(fields[i], std::make_pair(originRoi, targetRois[i]));
    }

    cv::Size collageSize(1056, 640);

    Collage<NameplateField> collage(image_, roiMapping, collageSize);
    std::vector<Detection> collageDets = detectorValuesStitched_.detect(collage.image());
    EnumHashMap<NameplateField, std::vector<Detection>> stitchedDets = collage.splitDetections(collageDets);
    postprocessStitchedDetections(stitchedDets);

    for (auto const& splitItem : stitchedDets) {
        NameplateField field = splitItem.first;
        std::vector<Detection> const& dets = splitItem.second;

        cv::Rect& originRoi = roiMapping[field].first;
        cv::Rect detsExtent = computeExtent(dets);
        std::cout << dets.front().rect << std::endl;
        // coordinates in dectections are relative to the source image
        // convert them to relative to the roi
        detsExtent = PerspectiveTransform(1, -originRoi.x, -originRoi.y).apply(detsExtent);
        adjustRoi(originRoi, detsExtent);
        OcrUtils::validateRoi(originRoi, image_);
    }

    collage = Collage<NameplateField>(image_, roiMapping, collageSize);
    collageDets = detectorValuesStitched_.detect(collage.image());
    stitchedDets = collage.splitDetections(collageDets);
    postprocessStitchedDetections(stitchedDets);

    for (auto& splitItem : stitchedDets) {
        updateByClassification(splitItem.second, image_);
    }

    for (auto const& splitItem : stitchedDets) {
        NameplateField field = splitItem.first;
        std::vector<Detection> const& dets = splitItem.second;

        auto itrKeyItem = keyOcrDetections_.find(field);
        if (itrKeyItem == keyOcrDetections_.end()) continue;
        OcrDetection const& keyItem = itrKeyItem->second;

        std::string valueText = joinDetectedChars(dets);
        cv::Rect const& valueRoi = roiMapping[field].first;
        OcrDetection valueItem(valueText, valueRoi);

        result_.emplace(field, KeyValueDetection(keyItem, valueItem));
    }
}

void OcrNameplatesAlfa::postprocessStitchedDetections(EnumHashMap<NameplateField, std::vector<Detection>>& stitchedDets) {
    for (auto& splitItem : stitchedDets) {
        NameplateField field = splitItem.first;
        std::vector<Detection>& dets = splitItem.second;

        sortByXMid(dets);
        eliminateXOverlaps(dets, field);
        eliminateYOutliers(dets);

        if (!shouldContainLetters(field)) {
            eliminateLetters(dets);
        }

        auto itrValueLength = VALUE_LENGTH.find(field);
        int valueLength = (itrValueLength == VALUE_LENGTH.end() ? 20 : itrValueLength->second);
        if (dets.size() > valueLength) {
            std::partial_sort(dets.begin(), std::next(dets.begin(), valueLength), dets.end(),
                              [](Detection const& lhs, Detection const& rhs) { return lhs.score > rhs.score; });
            dets.resize(valueLength);
            sortByXMid(dets);
        }
    }
}

bool OcrNameplatesAlfa::shouldContainLetters(NameplateField field) {
    switch (field) {
        case NameplateField::VIN:
        case NameplateField::VEHICLE_MODEL: {
            return true;
        }

        case NameplateField::ENGINE_MODEL:
        case NameplateField::MAX_MASS_ALLOWED:
        case NameplateField::MAX_NET_POWER_OF_ENGINE:
        case NameplateField::ENGINE_DISPLACEMENT:
        case NameplateField::DATE_OF_MANUFACTURE:
        case NameplateField::NUM_PASSENGERS:
        case NameplateField::PAINT: {
            return false;
        }

        default: throw std::invalid_argument("Input field unknown.");
    }
}

} // end namespace cuizhou