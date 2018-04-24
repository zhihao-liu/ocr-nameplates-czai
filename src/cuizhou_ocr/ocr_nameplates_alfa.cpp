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

std::vector<std::string> const OcrNameplatesAlfa::PAINT_CANDIDATES = {"414", "361", "217", "248", "092", "093", "035", "318", "408", "409", "620"};
EnumHashMap<OcrNameplatesAlfa::NameplateField, int> const OcrNameplatesAlfa::VALUE_LENGTH = {
        {NameplateField::VIN, 17},
        {NameplateField::MAX_MASS_ALLOWED, 4},
        {NameplateField::MAX_NET_POWER_OF_ENGINE, 3},
        {NameplateField::ENGINE_MODEL, 8},
        {NameplateField::NUM_PASSENGERS, 5},
        {NameplateField::VEHICLE_MODEL, 8},
        {NameplateField::ENGINE_DISPLACEMENT, 4},
        {NameplateField::DATE_OF_MANUFACTURE, 6},
        {NameplateField::PAINT, 3}
};

OcrNameplatesAlfa::OcrNameplatesAlfa(Detector const& detectorKeys,
                                     Detector const& detectorValuesVin,
                                     Detector const& detectorValuesOthers,
                                     Detector const& detectorValuesStitched,
                                     Classifier const& classifierChars)
        : detectorKeys_(detectorKeys),
          detectorValuesVin_(detectorValuesVin),
          detectorValuesOthers_(detectorValuesOthers),
          detectorValuesStitched_(detectorValuesStitched),
          classifierChars_(classifierChars) {}

void OcrNameplatesAlfa::processImage() {
    result_.clear();

    image_ = OcrUtils::imgResizeAndFill(image_, STANDARD_IMG_WIDTH, STANDARD_IMG_HEIGHT);

    detectKeys();
    adaptiveRotationWithUpdatingKeyDetections();
    
    // DEBUG
    auto d_stiched = stitchValueSubimgs();
    d_imgToShow = image_.clone();
    detectorValuesStitched_.setThresh(0.05, 0.3);
    auto d_dets = detectorValuesStitched_.detect(d_stiched.image());
    sortByXMid(d_dets);
    eliminateXOverlapsForOthers(d_dets);
    auto d_splitResults = d_stiched.splitDetections(d_dets);
    for (auto const& res : d_splitResults) {
        cv::Scalar clr(std::rand() % 256, std::rand() % 256, std::rand() % 256);
        for (auto const& det : res.second) {
            cv::rectangle(d_imgToShow, det.getRect(), clr, 2);
            std::cout << "RECT: " << det.getClass() << " -- " << det.getRect() << std::endl;
        }
    }
    std::cout << "DEBUG -- Dets Count: " << d_dets.size() << std::endl;
    // END_DEBUG

    for (auto const& keyItem : keyDetectedItems_) {
        DetectedItem valueItem;
        switch (keyItem.first) {
            case NameplateField::MANUFACTURER: {
                valueItem = DetectedItem("阿尔法 罗密欧股份公司", cv::Rect());
            } break;
            case NameplateField::BRAND: {
                valueItem = DetectedItem("阿尔法 罗密欧", cv::Rect());
            } break;
            case NameplateField::COUNTRY: {
                valueItem = DetectedItem("意大利", cv::Rect());
            } break;
            case NameplateField::FACTORY: {
                valueItem = DetectedItem("FCA意大利股份公司卡西诺工厂", cv::Rect());
            } break;
            default: {
                valueItem = detectValue(keyItem.first);
            } break;
        }

        result_.emplace(keyItem.first, KeyValuePair(keyItem.second, valueItem));
    }
}

void OcrNameplatesAlfa::detectKeys() {
    keyDetectedItems_.clear();

    detectorKeys_.setThresh(0.5, 0.1); // fixed params, empirical

    std::vector<Detection> keyDets = detectorKeys_.detect(image_);
    std::transform(keyDets.cbegin(), keyDets.cend(),
                   std::inserter(keyDetectedItems_, keyDetectedItems_.end()),
                   [](Detection const& det) {
                       std::string keyName = det.getClass();
                       DetectedItem keyItem(keyName, det.getRect());
                       return std::make_pair(fieldDict_.toEnum(keyName), keyItem);
                   });
}

DetectedItem OcrNameplatesAlfa::detectValue(NameplateField field) {
    switch (field) {
        case NameplateField::VIN: {
            return detectValueOfVin();
        }
        case NameplateField::MAX_MASS_ALLOWED:
        case NameplateField::DATE_OF_MANUFACTURE:
        case NameplateField::MAX_NET_POWER_OF_ENGINE:
        case NameplateField::ENGINE_MODEL:
        case NameplateField::NUM_PASSENGERS:
        case NameplateField::VEHICLE_MODEL:
        case NameplateField::ENGINE_DISPLACEMENT:
        case NameplateField::PAINT: {
            auto itrKeyItem = keyDetectedItems_.find(field);
            if (itrKeyItem == keyDetectedItems_.end()) return DetectedItem();

            detectorValuesVin_.setThresh(0.05, 0.3);

            cv::Rect valueRoi = estimateValueRoi(field, itrKeyItem->second.rect);
            valueRoi = OcrUtils::validateRoi(valueRoi, image_);
            cv::Mat valueSubimg = OcrUtils::imgResizeAndFill(image_(valueRoi), 1056, 640);
            std::vector<Detection> valueDets = detectorValuesOthers_.detect(valueSubimg);

            sortByXMid(valueDets);
            eliminateXOverlapsForVin(valueDets);
            eliminateYOutliers(valueDets);
            if (field != NameplateField::VEHICLE_MODEL) eliminateLetters(valueDets);

            std::string result = joinDetectedChars(valueDets);
            return DetectedItem(result, valueRoi);
        }
        default: return DetectedItem();
    }
}

void OcrNameplatesAlfa::adaptiveRotationWithUpdatingKeyDetections() {
    auto itrKeyVin = keyDetectedItems_.find(NameplateField::VIN);
    if (itrKeyVin == keyDetectedItems_.end()) return;

    detectorValuesVin_.setThresh(0.1, 0.3);

    cv::Rect keyRoi = itrKeyVin->second.rect;
    cv::Rect valueRoi = estimateValueRoi(NameplateField::VIN, keyRoi);
    std::vector<Detection> valueDets = detectorValuesVin_.detect(image_(valueRoi));

    sortByXMid(valueDets);
    eliminateXOverlapsForVin(valueDets);
    eliminateYOutliers(valueDets);

    double slope = estimateCharAlignmentSlope(valueDets);
    if (std::abs(slope) > 0.025) {
        double angle = std::atan(slope) / CV_PI * 180;
        image_ = OcrUtils::imgRotate(image_, angle);
        detectKeys(); // update detections of keys
    }
}

DetectedItem OcrNameplatesAlfa::detectValueOfVin() {
    auto itrKeyVin = keyDetectedItems_.find(NameplateField::VIN);
    if (itrKeyVin == keyDetectedItems_.end()) return DetectedItem();

    detectorValuesVin_.setThresh(0.05, 0.3);

    cv::Rect keyRoi = itrKeyVin->second.rect;
    cv::Rect valueRoi = estimateValueRoi(NameplateField::VIN, keyRoi);
    valueRoi = OcrUtils::validateRoi(valueRoi, image_);
    // no need to resize and fill because the model for VIN is trained with stretched images
    std::vector<Detection> valueDets = detectorValuesVin_.detect(image_(valueRoi));

    sortByXMid(valueDets);
    eliminateXOverlapsForVin(valueDets);
    eliminateYOutliers(valueDets);

    // second std::round in the network
    valueRoi = adjustRoi(valueRoi, computeExtent(valueDets));
    valueRoi = expandRoi(valueRoi, valueDets);
    valueRoi = OcrUtils::validateRoi(valueRoi, image_);
    valueDets = detectorValuesVin_.detect(image_(valueRoi));

    sortByXMid(valueDets);
    eliminateXOverlapsForVin(valueDets);
    eliminateYOutliers(valueDets);

    cv::Rect detsExtent = computeExtent(valueDets);
    if (isRoiTooLarge(valueRoi, detsExtent)) {
        // third std::round in the network
        valueRoi = adjustRoi(valueRoi, detsExtent);
        valueRoi = OcrUtils::validateRoi(valueRoi, image_);
        valueDets = detectorValuesVin_.detect(image_(valueRoi));

        sortByXMid(valueDets);
        eliminateXOverlapsForVin(valueDets);
        eliminateYOutliers(valueDets);
    }

    if (valueDets.size() < 17) {
        // fourth std::round in the network
        addGapDetections(valueDets, valueRoi);
        sortByXMid(valueDets);
    }

    if (valueDets.size() > 17) {
        mergeOverlappedDetections(valueDets);
    }

    reexamineCharsWithLowConfidence(valueDets, image_(valueRoi));

    std::string value = joinDetectedChars(valueDets);
    return DetectedItem(value, valueRoi);
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

Collage<OcrNameplatesAlfa::NameplateField> OcrNameplatesAlfa::stitchValueSubimgs() {
    std::vector<NameplateField> fields = {NameplateField::ENGINE_MODEL, NameplateField::VEHICLE_MODEL, NameplateField::MAX_MASS_ALLOWED, NameplateField::MAX_NET_POWER_OF_ENGINE, NameplateField::ENGINE_DISPLACEMENT, NameplateField::DATE_OF_MANUFACTURE, NameplateField::NUM_PASSENGERS, NameplateField::PAINT};
    std::vector<cv::Rect> targetRois = {cv::Rect(0, 0, 396, 320), cv::Rect(0, 320, 396, 320), cv::Rect(396, 0, 264, 320), cv::Rect(396, 320, 264, 320), cv::Rect(660, 0, 264, 320), cv::Rect(660, 320, 264, 320), cv::Rect(924, 0, 132, 320), cv::Rect(924, 320, 132, 320)};

    std::vector<cv::Rect> originRois;
    std::transform(fields.cbegin(), fields.cend(), std::back_inserter(originRois),
                   [&](NameplateField field) {
                        auto itrKeyItem = keyDetectedItems_.find(field);
                        cv::Rect valueRoi;
                        if (itrKeyItem != keyDetectedItems_.end()) {
                            valueRoi = estimateValueRoi(field, itrKeyItem->second.rect);
                            valueRoi = OcrUtils::validateRoi(valueRoi, image_);
                        }
                        return valueRoi;
    });


    return Collage<NameplateField>(image_, fields, originRois, targetRois, cv::Size(1056, 640));
}

void OcrNameplatesAlfa::sortByXMid(std::vector<Detection>& dets) {
    std::sort(dets.begin(), dets.end(), [](Detection const& det1, Detection const& det2) {
        return OcrUtils::xMid(det1.getRect()) < OcrUtils::xMid(det2.getRect());
    });
}

void OcrNameplatesAlfa::sortByYMid(std::vector<Detection>& dets) {
    std::sort(dets.begin(), dets.end(), [](Detection const& det1, Detection const& det2) {
        return OcrUtils::yMid(det1.getRect()) < OcrUtils::yMid(det2.getRect());
    });
}

void OcrNameplatesAlfa::sortByScoreDescending(std::vector<Detection>& dets) {
    std::sort(dets.begin(), dets.end(),
              [](Detection const& det1, Detection const& det2) { return det1.getScore() > det2.getScore(); });
}

bool OcrNameplatesAlfa::isSortedByXMid(std::vector<Detection> const& dets) {
    return std::is_sorted(dets.cbegin(), dets.cend(),
                          [](Detection const& det1, Detection const& det2) {
                              return OcrUtils::xMid(det1.getRect()) < OcrUtils::xMid(det2.getRect());
                          });
}

std::string OcrNameplatesAlfa::joinDetectedChars(std::vector<Detection> const& dets) {
    assert(isSortedByXMid(dets));

    return std::accumulate(dets.cbegin(), dets.cend(), std::string(),
                           [](std::string const& res, Detection const& det) { return res + det.getClass(); });
}

bool OcrNameplatesAlfa::containsUnambiguousNumberOne(Detection const& det1, Detection const& det2) {
    return (det1.getClass() == "1" && det2.getClass() != "7" && det2.getClass() != "4" && det2.getClass() != "H")
           || (det2.getClass() == "1" && det1.getClass() != "7" && det1.getClass() != "4" && det1.getClass() != "H");
}

void OcrNameplatesAlfa::eliminateYOutliers(std::vector<Detection>& dets) {
    if (dets.size() < 3) return;

    // remove those lies off the horizontal reference line
    int heightRef = OcrUtils::findMedian(dets, [](Detection const& det) { return det.getRect().height; });

    int yMidRef = OcrUtils::findMedian(dets, [](Detection const& det) { return OcrUtils::yMid(det.getRect()); });

    for (auto itr = dets.begin(); itr != dets.end();) {
        if (std::abs(OcrUtils::yMid(itr->getRect()) - yMidRef) > 0.25 * heightRef) {
            itr = dets.erase(itr);
        } else {
            ++itr;
        }
    }
}

void OcrNameplatesAlfa::eliminateXOverlapsForVin(std::vector<Detection>& dets) {
    eliminateXOverlapsWithThresh(dets, {0.6, 0.4}, {0.6, 0.3});
}

void OcrNameplatesAlfa::eliminateXOverlapsForOthers(std::vector<Detection>& dets) {
    eliminateXOverlapsWithThresh(dets, {0.5, 0.3}, {0.5, 0.2});
}

void OcrNameplatesAlfa::eliminateXOverlapsWithThresh(std::vector<Detection>& dets, std::pair<float, float> firstThresh, std::pair<float, float> secondThresh) {
    if (dets.size() < 2) return;
    assert(isSortedByXMid(dets));

    for (auto itr = std::next(dets.begin()); itr != dets.end();) {
        // set larger tolerance for "1" because it is overlapped most of the time
        float firstTol = containsUnambiguousNumberOne(*std::prev(itr), *itr) ? firstThresh.first : firstThresh.second;
        float secondTol = containsUnambiguousNumberOne(*std::prev(itr), *itr) ? secondThresh.first : secondThresh.second;

        float overlap = OcrUtils::computeIou(std::prev(itr)->getRect(), itr->getRect());
        if (overlap > firstTol) {
            if (std::prev(itr)->getScore() < itr->getScore()) {
                itr = std::next(dets.erase(std::prev(itr)));
                continue;
            } else {
                itr = dets.erase(itr);
                continue;
            }
        } else if (overlap > secondTol) {
            if (std::prev(itr)->getScore() < itr->getScore() && std::prev(itr)->getScore() < 0.2) {
                itr = std::next(dets.erase(std::prev(itr)));
                continue;
            } else if (itr->getScore() < std::prev(itr)->getScore() && itr->getScore() < 0.2) {
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
        left = std::min(left, det.getRect().x);
        right = std::max(right, det.getRect().br().x);
        top = std::min(top, det.getRect().y);
        bottom = std::max(bottom, det.getRect().br().y);
    }

    return cv::Rect(left, top, right - left + 1, bottom - top + 1);
}

double OcrNameplatesAlfa::estimateCharAlignmentSlope(std::vector<Detection> const& dets) {
    if (dets.empty()) return 0;

    std::vector<double> xCoords, yCoords;
    std::transform(dets.cbegin(), dets.cend(), std::back_inserter(xCoords),
                   [](Detection det) { return OcrUtils::xMid(det.getRect()); });
    std::transform(dets.cbegin(), dets.cend(), std::back_inserter(yCoords),
                   [](Detection det) { return OcrUtils::yMid(det.getRect()); });

    LeastSquare ls(xCoords, yCoords);
    return ls.getSlope();
}

cv::Rect OcrNameplatesAlfa::expandRoi(cv::Rect const& roi, std::vector<Detection> const& dets) {
    assert(isSortedByXMid(dets));

    int vacancy = 17 - int(dets.size());
    if (vacancy <= 0) return roi;

    float charWidth = float(computeExtent(dets).width) / dets.size();
    float additionalWidth = float(charWidth * vacancy * 1.05);

    int newX = roi.x;
    int newY = roi.y;
    int newW = int(std::round(roi.width + additionalWidth));
    int newH = roi.height;

    double slope = estimateCharAlignmentSlope(dets);

    if (slope > 0) {
        newH += std::round(slope * additionalWidth);
    } else {
        newY += std::round(slope * additionalWidth);
    }

    return cv::Rect(newX, newY, newW, newH);
}

bool OcrNameplatesAlfa::isRoiTooLarge(cv::Rect const& roi, cv::Rect const& detsExtent) {
    return (roi.width - detsExtent.width > 2.5 * ROI_X_BORDER) ||
           (roi.height - detsExtent.height > 2.5 * ROI_Y_BORDER);
}

cv::Rect OcrNameplatesAlfa::adjustRoi(cv::Rect const& roi, cv::Rect const& detsExtent) {
    int newLeft = roi.x + (detsExtent.x - ROI_X_BORDER);
    int newRight = roi.x + (detsExtent.br().x + ROI_X_BORDER);
    int newTop = roi.y + (detsExtent.y - ROI_Y_BORDER);
    int newBottom = roi.y + (detsExtent.br().y + ROI_Y_BORDER);

    return cv::Rect(newLeft, newTop, newRight - newLeft + 1, newBottom - newTop + 1);
}

int OcrNameplatesAlfa::estimateCharSpacing(std::vector<Detection> const& dets) {
    if (dets.size() < 2) return 0;
    assert(isSortedByXMid(dets));

    std::vector<int> spacings;
    for (auto itr = std::next(dets.cbegin()); itr != dets.cend(); ++itr) {
        int spacing = OcrUtils::computeSpacing(std::prev(itr)->getRect(), itr->getRect());
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
        cv::Rect leftRect = std::prev(itr)->getRect();
        cv::Rect rightRect = itr->getRect();
        if (OcrUtils::computeSpacing(leftRect, rightRect) > 1.5 * spacingRef) {
            int gapX = leftRect.x + leftRect.width - CHAR_X_BORDER;
            int gapY = (leftRect.y + rightRect.y) / 2 - CHAR_Y_BORDER;
            int gapW = (rightRect.x - (leftRect.x + leftRect.width)) + CHAR_X_BORDER * 2;
            int gapH = (leftRect.height + rightRect.height) / 2 + CHAR_Y_BORDER * 2;

            int gapXReal = roi.x + gapX;
            int gapYReal = roi.y + gapY;

            cv::Rect gapRect(gapXReal, gapYReal, gapW, gapH);
            gapRect = OcrUtils::validateRoi(gapRect, image_);

            cv::Mat gapExpanded(gapRect.height, gapRect.width * 10, CV_8UC3, cv::Scalar(0, 0, 0));
            cv::Rect centerRegion(cv::Point(gapExpanded.cols / 2 - gapRect.width / 2, 0), gapRect.size());

            image_(gapRect).copyTo(gapExpanded(centerRegion));
            std::vector<Detection> gapDets = detectorValuesVin_.detect(gapExpanded);

            if (!gapDets.empty()) {
                gapRect.x = gapX;
                gapRect.y = gapY;
                gapRect = OcrUtils::validateRoi(gapRect, roi);

                Detection gapDet = gapDets.front();
                gapDet.setRect(gapRect);

                addedDets.push_back(gapDet);
            }
        }
    }

    dets.insert(dets.end(), addedDets.cbegin(), addedDets.cend());
}

void OcrNameplatesAlfa::reexamineCharsWithLowConfidence(std::vector<Detection>& dets, cv::Mat const& subimg) const {
    for (auto& det : dets) {
        if (!OcrUtils::isNumbericChar(det.getClass())) continue;
        if (det.getScore() >= 0.8) continue;

        vector<Prediction> pres = classifierChars_.classify(subimg(det.getRect()), 1);
        if (pres[0].second > 0.9) {
            det.setClass(pres[0].first);
            det.setScore(pres[0].second);
        }
    }
};

void OcrNameplatesAlfa::mergeOverlappedDetections(std::vector<Detection>& dets) {
    if (dets.size() <= 17) return;

    std::vector<float> overlaps;
    for (auto itr = std::next(dets.cbegin()); itr != dets.cend(); ++itr) {
        float iou = OcrUtils::computeIou(std::prev(itr)->getRect(), itr->getRect());
        overlaps.push_back(iou);
    }

    int idxMaxOverlap = std::distance(overlaps.cbegin(), std::max_element(overlaps.cbegin(), overlaps.cend()));

    int idxToErase = dets[idxMaxOverlap].getScore() < dets[idxMaxOverlap + 1].getScore() ? idxMaxOverlap : idxMaxOverlap + 1;
    dets.erase(std::next(dets.begin(), idxToErase));

    mergeOverlappedDetections(dets);
}

void OcrNameplatesAlfa::eliminateLetters(std::vector<Detection>& dets) {
    for (auto itr = dets.begin(); itr != dets.end(); ) {
        std::string fallback = fallbackToNumber(itr->getClass());
        if (fallback.empty()) {
            itr = dets.erase(itr);
        } else {
            (itr++)->setClass(fallback);
        }
    }
}

std::string OcrNameplatesAlfa::fallbackToNumber(std::string const& str) {
    if (str.length() != 1) return std::string();

    switch(str[0]) {
        case '0' ... '9': return str;
        case 'A': return "4";
        case 'C': return "0";
        case 'I': return "1";
        case 'O': return "0";
        case 'S': return "5";
        case 'Z': return "2";
        default: return std::string();
    }
}

std::string OcrNameplatesAlfa::matchPaintWithLengthFixed(std::string const& str) {
    assert(str.length() == 3);
    for (auto const& cand : PAINT_CANDIDATES) {
        bool matched1 = (str[0] == '$' || str[0] == cand[0]);
        bool matched2 = (str[1] == '$' || str[1] == cand[1]);
        bool matched3 = (str[2] == '$' || str[2] == cand[2]);
        if (matched1 && matched2 && matched3) return cand;
    }
    return std::string();
}

std::string OcrNameplatesAlfa::matchPaint(std::string const& str) {
    std::string result;

    if (str.length() > 3) {
        return matchPaint(str.substr(0, 3));
    } else if (str.length() == 3) {
        if (result.empty()) result = matchPaintWithLengthFixed(str);
        if (result.empty()) result = matchPaintWithLengthFixed(str.substr(0, 2) + "$");
        if (result.empty()) result = matchPaintWithLengthFixed(str.substr(0, 1) + "$" + str.substr(2, 1));
        if (result.empty()) result = matchPaintWithLengthFixed("$" + str.substr(1, 2));
    } else if (str.length() == 2) {
        if (result.empty()) result = matchPaintWithLengthFixed(str + "$");
        if (result.empty()) result = matchPaintWithLengthFixed(str.substr(0, 1) + "$" + str.substr(1, 1));
        if (result.empty()) result = matchPaintWithLengthFixed("$" + str);
    }

    if (result.empty()) result = str;
    return result;
}

DetectedItem OcrNameplatesAlfa::detectValueOfOthers(NameplateField field) {

}


// -------- added by WRZ -------- //

DetectedItem OcrNameplatesAlfa::detectValueOfMaxMassAllowed() {
    auto itrKey = keyDetectedItems_.find(NameplateField::MAX_MASS_ALLOWED);
    if (itrKey == keyDetectedItems_.end()) return DetectedItem();

    detectorValuesOthers_.setThresh(0.05, 0.3);
    cv::Rect keyRoi = itrKey->second.rect;

    int x, y, w, h;
    x = keyRoi.x + keyRoi.width;
    y = keyRoi.y + 3;
    w = std::max(int(keyRoi.width * 0.1), 95);
    h = std::max(int(keyRoi.height * 1), 45);
    cv::Rect roi(x, y, std::max(0, w), std::max(0, h));
    roi = OcrUtils::validateRoi(roi, image_);

    std::string valueContent;
    int valueLength = 4;
    commonDetectProcess(valueContent, detectorValuesOthers_, image_, roi, valueLength);

    return DetectedItem(valueContent, roi);
}

DetectedItem OcrNameplatesAlfa::detectValueOfDateOfManufacture() {
    auto itrKey = keyDetectedItems_.find(NameplateField::DATE_OF_MANUFACTURE);
    if (itrKey == keyDetectedItems_.end()) return DetectedItem();

    detectorValuesOthers_.setThresh(0.05, 0.3);
    cv::Rect keyRoi = itrKey->second.rect;

    string result;
    int x, y, w, h;
    x = keyRoi.x + keyRoi.width + 5;
    y = keyRoi.y - 2;
    w = std::max(int(keyRoi.width * 1.5), 150);
    h = int(keyRoi.height * 1.2);
    cv::Rect roi(x, y, std::max(0, w), std::max(0, h));
    roi = OcrUtils::validateRoi(roi, image_);

    int valueLength = 6;
    commonDetectProcess(result, detectorValuesOthers_, image_, roi, valueLength);
    return DetectedItem(result, roi);
}

DetectedItem OcrNameplatesAlfa::detectValueOfMaxNetPowerOfEngine() {
    auto itrKey = keyDetectedItems_.find(NameplateField::MAX_NET_POWER_OF_ENGINE);
    if (itrKey == keyDetectedItems_.end()) return DetectedItem();

    detectorValuesOthers_.setThresh(0.05, 0.3);
    cv::Rect keyRoi = itrKey->second.rect;

    string result;
    int x, y, w, h;
    x = keyRoi.x + keyRoi.width;
    y = keyRoi.y + 10;
    w = std::max(int(keyRoi.width * 0.1), 90);
    h = std::max(int(keyRoi.height * 1), 40);
    cv::Rect roi(x, y, std::max(0, w), std::max(0, h));
    roi = OcrUtils::validateRoi(roi, image_);

    int valueLength = 3;
    commonDetectProcess(result, detectorValuesOthers_, image_, roi, valueLength);

    if (result.empty()) {
        x = keyRoi.x + keyRoi.width;
        y = keyRoi.y - 5;
        w = std::max(int(keyRoi.width * 0.1), 90);
        h = std::max(int(keyRoi.height * 1), 40);
        roi = cv::Rect(x, y, std::max(0, w), std::max(0, h));
        commonDetectProcess(result, detectorValuesOthers_, image_, roi, valueLength);
    }

    for (int i = 0; i < result.size(); ++i) {
        if (result.substr(i, 1) == "5" || result.substr(i, 1) == "3" || result.substr(i, 1) == "6" ||
            result.substr(i, 1) == "2" || result.substr(i, 1) == "0") {
            result = "206";
        } else if (result.substr(i, 1) == "1" || result.substr(i, 1) == "4" || result.substr(i, 1) == "7") {
            result = "147";
        }
    }

    return DetectedItem(result, roi);
}

DetectedItem OcrNameplatesAlfa::detectValueOfNumPassengers() {
    auto itrKey = keyDetectedItems_.find(NameplateField::NUM_PASSENGERS);
    if (itrKey == keyDetectedItems_.end()) return DetectedItem();

    detectorValuesVin_.setThresh(0.2, 0.1);
    cv::Rect keyRoi = itrKey->second.rect;

    string result;
    int x, y, w, h;
    x = keyRoi.x + keyRoi.width + 5;
    y = keyRoi.y;
    w = std::max(int(keyRoi.width * 0.1), 45);
    h = std::max(int(keyRoi.height * 1), 40);
    cv::Rect roi(x, y, std::max(0, w), std::max(0, h));
    roi = OcrUtils::validateRoi(roi, image_);

    cv::Mat subimg;
    image_(roi).copyTo(subimg);

    vector<Detection> valueDets = detectorValuesVin_.detect(subimg);
    sortByXMid(valueDets);
    eliminateXOverlapsForVin(valueDets);

    vector<Detection> temp1;
    for (auto const& det : valueDets) {
        string cls = det.getClass();
        if (OcrUtils::isNumbericChar(cls)) temp1.push_back(det);
    }

    vector<Detection> temp2;
    if (temp1.size() > 1) {
        sortByScoreDescending(temp1);
        temp2.insert(temp2.end(), temp1.begin(), std::next(temp1.begin()));
        sortByXMid(temp2);
        for (auto const& det : temp2) {
            result.append(det.getClass());
        }
    } else {
        temp2 = temp1;
        for (auto const& det : temp2) {
            result.append(det.getClass());
        }
    }

    return DetectedItem(result, roi);
}

DetectedItem OcrNameplatesAlfa::detectValueOfEngineModel() {
    auto itrKey = keyDetectedItems_.find(NameplateField::ENGINE_MODEL);
    if (itrKey == keyDetectedItems_.end()) return DetectedItem();

    detectorValuesOthers_.setThresh(0.05, 0.3);
    cv::Rect keyRoi = itrKey->second.rect;

    string result;
    int x, y, w, h;
    x = keyRoi.x + keyRoi.width + 5;
    y = keyRoi.y;
    w = std::max(int(keyRoi.width * 0.1), 150);
    h = std::max(int(keyRoi.height * 1), 40);
    cv::Rect roi(x, y, std::max(0, w), std::max(0, h));
    roi = OcrUtils::validateRoi(roi, image_);

    int valueLength = 8;
    commonDetectProcess(result, detectorValuesOthers_, classifierChars_, image_, roi, valueLength);

    return DetectedItem(result, roi);
}

DetectedItem OcrNameplatesAlfa::detectValueOfVehicleModel() {
    auto itrKey = keyDetectedItems_.find(NameplateField::VEHICLE_MODEL);
    if (itrKey == keyDetectedItems_.end()) return DetectedItem();

    detectorValuesOthers_.setThresh(0.05, 0.3);
    cv::Rect keyRoi = itrKey->second.rect;

    string result;
    int x, y, w, h;
    x = std::min(keyRoi.x + keyRoi.width + 5, image_.cols - 1);
    y = keyRoi.y;
    w = std::max(std::min(int(image_.cols - 1 - x), 120), 0);
    h = std::max(int(keyRoi.height * 1), 40);
    cv::Rect roi(x, y, std::max(0, w), std::max(0, h));
    roi = OcrUtils::validateRoi(roi, image_);

    int valueLength = 8;
    bool containEnglishChar = true;
    commonDetectProcessForVehicleModel(result, detectorValuesOthers_, image_, roi, valueLength, containEnglishChar);

    return DetectedItem(result, roi);
}

DetectedItem OcrNameplatesAlfa::detectValueOfEngineDisplacement() {
    auto itrKey = keyDetectedItems_.find(NameplateField::ENGINE_DISPLACEMENT);
    if (itrKey == keyDetectedItems_.end()) return DetectedItem();

    detectorValuesOthers_.setThresh(0.05, 0.3);
    cv::Rect keyRoi = itrKey->second.rect;

    string result;
    int x, y, w, h;
    x = keyRoi.x + keyRoi.width + 5;
    y = keyRoi.y;//keyRoi.y-5;
    w = std::max(int(keyRoi.width * 0.1), 100);
    h = std::max(int(keyRoi.height * 1), 35);
    cv::Rect roi(x, y, std::max(0, w), std::max(0, h));
    roi = OcrUtils::validateRoi(roi, image_);

    int valueLength = 4;
    commonDetectProcess(result, detectorValuesOthers_, image_, roi, valueLength);
    return DetectedItem(result, roi);
}


DetectedItem OcrNameplatesAlfa::detectValueOfPaint() {
    auto itrKey = keyDetectedItems_.find(NameplateField::PAINT);
    if (itrKey == keyDetectedItems_.end()) return DetectedItem();

    detectorValuesOthers_.setThresh(0.05, 0.3);
    cv::Rect keyRoi = itrKey->second.rect;

    string result;
    int x, y, w, h;
    x = keyRoi.x + keyRoi.width;
    y = keyRoi.y;
    w = std::max(int(keyRoi.width * 0.1), 45);
    h = std::max(int(keyRoi.height * 1), 30);
    cv::Rect roi(x, y, std::max(0, w), std::max(0, h));
    roi = OcrUtils::validateRoi(roi, image_);

    int valueLength = 3;
    commonDetectProcess(result, detectorValuesOthers_, image_, roi, valueLength); // thresh = 0.05?
    result = matchPaint(result);

    return DetectedItem(result, roi);
}

void OcrNameplatesAlfa::commonDetectProcess(string& result, Detector& detectorValues, cv::Mat const& img,
                                            cv::Rect const& roi, int valueLength, bool containsLetters,
                                            float confThresh, float iouThresh) {
    result.clear();

    vector<Detection> temp1;
    subCommonDetectProcess(detectorValues, img, roi, temp1, containsLetters, confThresh, iouThresh);

    if (temp1.empty()) return;

    sortByXMid(temp1);
    vector<Detection> temp2;
    bool isMoved = false;
    cv::Rect roiMoved;

    if (temp1.size() == valueLength) {
        temp2 = temp1;
        for (auto const& det : temp2) {
            if (det.getClass() == "Z") result.append("7");
            else result.append(det.getClass());
        }
    } else {
        isMoved = moveRoi(img, temp1, roi, roiMoved);

        subCommonDetectProcess(detectorValues, img, roiMoved, temp2, containsLetters, confThresh, iouThresh);

        if (temp2.size() > valueLength) {
            sortByScoreDescending(temp2);
            vector<Detection> temp3;
            temp3.insert(temp3.end(), temp2.begin(), temp2.begin() + valueLength);
            sortByXMid(temp3);
            for (auto const& det : temp3) {
                if (det.getClass() == "Z") result.append("7");
                else result.append(det.getClass());
            }
            temp2.clear();
            temp2.insert(temp2.end(), temp3.begin(), temp3.end());
        } else if (temp2.size() <= valueLength) {
            sortByXMid(temp2);
            for (auto const& det : temp2) {
                if (det.getClass() == "Z") result.append("7");
                else result.append(det.getClass());
            }
        }
    }
}

void OcrNameplatesAlfa::commonDetectProcess(string& result, Detector& detectorValues, Classifier& classifier,
                                            cv::Mat const& img, cv::Rect const& roi, int valueLength,
                                            bool containEnglish, float conf, float iouThresh) {
    result.clear();

    vector<Detection> temp1;
    subCommonDetectProcess(detectorValues, img, roi, temp1, containEnglish, conf, iouThresh);

    if (temp1.empty()) return;

    sortByXMid(temp1);
    vector<Detection> temp2;
    bool isMoved = false;
    cv::Rect roiMoved;

    if (temp1.size() == valueLength) {
        temp2 = temp1;
        for (auto const& det : temp2) {
            if (det.getClass() == "Z") result.append("7");
            else result.append(det.getClass());
        }
    } else {
        isMoved = moveRoi(img, temp1, roi, roiMoved);

        subCommonDetectProcess(detectorValues, img, roiMoved, temp2, containEnglish, conf, iouThresh);

        if (temp2.size() > valueLength) {
            sortByScoreDescending(temp2);
            vector<Detection> temp3;
            temp3.insert(temp3.end(), temp2.begin(), temp2.begin() + valueLength);
            temp2.clear();
            temp2.insert(temp2.end(), temp3.begin(), temp3.end());
        }
    }


    result.clear();

    cv::Mat subimg;
    img(roi).copyTo(subimg);
    cropImg(subimg);
    if (isMoved) {
        subimg = img(roiMoved).clone();
        cropImg(subimg);
    }

    sortByXMid(temp2);
    for (auto const& det : temp2) {
        cv::Mat cropImg = subimg(det.getRect()).clone();
        cv::resize(cropImg, cropImg, cv::Size(224, 224));
        vector<Prediction> pre = classifier.classify(cropImg, 1);
        if (det.getScore() < 0.99) {
            if (pre[0].second < 0.85 || !OcrUtils::isNumbericChar(pre[0].first)) {
                result.append(det.getClass());
            } else {
                result.append(pre[0].first);
            }
        } else {
            result.append(det.getClass());
        }
    }
}

void OcrNameplatesAlfa::commonDetectProcessForVehicleModel(string& result, Detector& detectorValues,
                                                           cv::Mat const& img, cv::Rect const& roi, int valueLength,
                                                           bool containsLetters, float conf, float iouThresh) {
    result.clear();
    vector<Detection> temp1;
    subCommonDetectProcess(detectorValues, img, roi, temp1, containsLetters, conf, iouThresh);

    if (temp1.empty()) return;

    sortByXMid(temp1);
    eliminateXOverlapsForVin(temp1);

    vector<Detection> temp2;
    bool isMoved = false;
    cv::Rect roiMoved;

    if (temp1.size() == valueLength) {
        temp2 = temp1;
        for (auto const& det : temp2) {
            if (det.getClass() == "Z") result.append("7");
            else result.append(det.getClass());
        }
    } else {
        isMoved = moveRoi(img, temp1, roi, roiMoved);
        subCommonDetectProcess(detectorValues, img, roiMoved, temp2, containsLetters, conf, iouThresh);

        if (temp2.size() > valueLength) {
            sortByScoreDescending(temp2);
            vector<Detection> temp3;
            temp3.insert(temp3.end(), temp2.begin(), temp2.begin() + valueLength);
            sortByXMid(temp3);
            for (int i = 0; i < valueLength; ++i) {
                if (temp3[i].getClass() == "Z") result.append("7");
                else result.append(temp3[i].getClass());
            }
            temp2.clear();
            temp2.insert(temp2.end(), temp3.begin(), temp3.end());
        } else if (temp2.size() <= valueLength) {
            sortByXMid(temp2);
            for (auto const& det : temp2) {
                if (det.getClass() == "Z") result.append("7");
                else result.append(det.getClass());
            }
        }
    }
}

void OcrNameplatesAlfa::subCommonDetectProcess(Detector& detectorValues, cv::Mat const& img, cv::Rect const& roi,
                                               vector<Detection>& dets,
                                               bool containsLetters, float conf, float iouThresh) {
    cv::Mat subimg;
    img(roi).copyTo(subimg);
    cropImg(subimg);

    detectorValues.setThresh(conf, iouThresh);
    vector<Detection> valueDets = detectorValues.detect(subimg);

    vector<Detection> temp;
    if (!containsLetters) {
        for (Detection const& det: valueDets) {
            string cls = det.getClass();
            if (OcrUtils::isNumbericChar(cls)) temp.push_back(det);
        }
    } else {
        temp = valueDets;
    }

    sortByXMid(temp);
    eliminateXOverlapsForVin(temp);
    eliminateYOutliers(temp);

    dets.clear();
    dets.insert(dets.end(), temp.begin(), temp.end());
}

bool OcrNameplatesAlfa::moveRoi(cv::Mat const& img, vector<Detection>& dets, cv::Rect const& roi,
                                cv::Rect& newRoi) {
    int threshy = 100;
    bool needmove = false;
    newRoi = roi;

    if (dets.empty()) return false;
    int cy = 0;
    for (Detection const& det:dets) {
        if (det.getRect().y < threshy || std::abs(det.getRect().y + det.getRect().height - 640) < threshy) {
            needmove = true;
            cy = det.getRect().y + det.getRect().height / 2;
        }
    }
    if (needmove) {
        float ratio;
        if (float(roi.width) / roi.height > 1056.0 / 640) ratio = float(1056) / roi.width;
        else ratio = float(640) / roi.height;
        newRoi.y = roi.y + int((cy - 320) / ratio);
    }
    return needmove;
}

void OcrNameplatesAlfa::cropImg(cv::Mat& input) {
    cv::Mat subimg(642, 1058, CV_8UC3, cv::Scalar(0, 0, 0));
    int w = input.cols;
    int h = input.rows;
    int nw, nh;
    float ratio = 0.0;
    if (float(w) / h < 1056.0 / 640) {
        // h is main
        ratio = float(640) / h;
        nh = 640;
        nw = int(ratio * w);
        cv::resize(input, input, cv::Size(nw, nh));
        cv::Rect roi(528 - nw / 2, 0, nw, nh);
        input.copyTo(subimg(roi));
    } else {
        // w is main
        ratio = float(1056) / w;
        nh = int(h * ratio);
        nw = 1056;
        cv::resize(input, input, cv::Size(nw, nh));
        cv::Rect roi(0, std::max(0, 320 - nh / 2), nw, nh);
        input.copyTo(subimg(roi));
    }
    subimg.copyTo(input);
}

} // end namespace cuizhou