//
// Created by Zhihao Liu on 18-4-4.
//

#include "OcrNameplatesAlfa.h"
#include "OcrUtils.hpp"


using namespace cuizhou;

int const OcrNameplatesAlfa::WINDOW_X_BORDER = 8;
int const OcrNameplatesAlfa::WINDOW_Y_BORDER = 4;
int const OcrNameplatesAlfa::CHAR_X_BORDER = 2;
int const OcrNameplatesAlfa::CHAR_Y_BORDER = 1;

OcrNameplatesAlfa::~OcrNameplatesAlfa() = default;

OcrNameplatesAlfa::OcrNameplatesAlfa() = default;

OcrNameplatesAlfa::OcrNameplatesAlfa(PvaDetector& detectorKeys, PvaDetector& detectorValues1, PvaDetector& detectorValues2, Classifier& classifierChars)
        : _pDetectorKeys(&detectorKeys), _pDetectorValues1(&detectorValues1), _pDetectorValues2(&detectorValues2), _pClassifierChars(&classifierChars) {}

void OcrNameplatesAlfa::processImage() {
    _result.clear();

    OcrUtils::imResizeAndFill(_image, 1024, 768);

    detectKeys();

    for (auto const& keyItem: _keyDetectedItems) {
        if (keyItem.first == "Manufacturer") {
            _result.put(keyItem.first, KeyValuePair({keyItem.second, {"阿尔法 罗密欧股份公司"}}));
            continue;
        } else if (keyItem.first == "Brand") {
            _result.put(keyItem.first, KeyValuePair({keyItem.second, {"阿尔法 罗密欧"}}));
            continue;
        } else if (CLASSNAME_ENG_TO_CHN.find(keyItem.first)->second == "Country") {
            _result.put(keyItem.first, KeyValuePair({keyItem.second, {"意大利"}}));
            continue;
        } else if (keyItem.first == "Factory") {
            _result.put(keyItem.first, KeyValuePair({keyItem.second, {"FCA意大利股份公司卡西诺工厂"}}));
            continue;
        }

        DetectedItem valueItem = detectValue(keyItem.first);
        _result.put(keyItem.first, KeyValuePair({keyItem.second, valueItem}));
    }

//    cv::imshow("", _image);
//    cv::waitKey(0);
}

void OcrNameplatesAlfa::detectKeys() {
    _keyDetectedItems.clear();

    _pDetectorKeys->setThresh(0.5, 0.1);

    std::vector<Detection> keyDets = _pDetectorKeys->detect(_image);
    for (auto const& keyDet: keyDets) {
        std::string keyName = keyDet.getClass();
        DetectedItem detectedItem = {keyName, keyDet.getRect()};
        _keyDetectedItems.emplace(keyName, detectedItem);
    }
}

DetectedItem OcrNameplatesAlfa::detectValue(std::string const& keyName) {
    if (keyName == CLASSNAME_VIN) {
        return detectValueOfVin();
    } else if (keyName == CLASSNAME_MAX_MASS_ALLOWED) {
        return detectValueOfMaxMassAllowed();
    } else if (keyName == CLASSNAME_DATE_OF_MANUFACTURE) {
        return detectValueOfDateOfManufacture();
    } else if (keyName == CLASSNAME_MAX_NET_POWER_OF_ENGINE) {
        return detectValueOfMaxNetPowerOfEngine();
    } else if (keyName == CLASSNAME_ENGINE_MODEL) {
        return detectValueOfEngineModel();
    } else if (keyName == CLASSNAME_NUM_PASSENGERS) {
        return detectValueOfNumPassengers();
    } else if (keyName == CLASSNAME_VEHICLE_MODEL) {
        return detectValueOfVehicleModel();
    } else if (keyName == CLASSNAME_ENGINE_DISPLACEMENT) {
        return detectValueOfEngineDisplacement();
    } else if (keyName == CLASSNAME_PAINT) {
        return detectValueOfPaint();
    }

    return DetectedItem();
}

DetectedItem OcrNameplatesAlfa::detectValueOfVin() {
    auto itrKeyVin = _keyDetectedItems.find(CLASSNAME_VIN);
    if (itrKeyVin == _keyDetectedItems.end()) return DetectedItem();

    _pDetectorValues1->setThresh(0.05, 0.3);

    cv::Rect keyRect = itrKeyVin->second.rect;
    cv::Rect valueRect = estimateValueRectOfVin(keyRect);
    std::vector<Detection> valueDets = _pDetectorValues1->detect(_image(valueRect));

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
        valueDets = _pDetectorValues1->detect(_image(valueRect));

        sortByXMid(valueDets);
        eliminateYOutliers(valueDets);
        eliminateOverlaps(valueDets);
    }

    // second round in the network
    adjustWindow(valueRect, computeExtent(valueDets));
    expandWindow(valueRect, valueDets);
    OcrUtils::validateWindow(valueRect, _image);
    valueDets = _pDetectorValues1->detect(_image(valueRect));

    sortByXMid(valueDets);
    eliminateYOutliers(valueDets);
    eliminateOverlaps(valueDets);

    cv::Rect detsExtent = computeExtent(valueDets);
    if (isWindowTooLarge(valueRect, detsExtent)) {
        // third round in the network
        adjustWindow(valueRect, detsExtent);
        OcrUtils::validateWindow(valueRect, _image);
        valueDets = _pDetectorValues1->detect(_image(valueRect));

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

    reexamineCharsWithLowConfidence(valueDets, _image(valueRect));

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

void OcrNameplatesAlfa::sortByYMid(std::vector<Detection>& dets) {
    std::sort(dets.begin(), dets.end(), [](Detection const& det1, Detection const& det2) {
        return OcrUtils::yMid(det1.getRect()) < OcrUtils::yMid(det2.getRect());
    });
}

void OcrNameplatesAlfa::sortByScoreDescending(std::vector<Detection>& dets) {
    std::sort(dets.begin(), dets.end(), [](Detection const& det1, Detection const& det2) {
        return det1.getScore() > det2.getScore();
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
    int heightRef = int(OcrUtils::findMedian(dets, [](Detection const& det){ return det.getRect().height; }));

    int yMidRef = int(OcrUtils::findMedian(dets, [](Detection const& det){ return OcrUtils::yMid(det.getRect()); }));

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

cv::Rect& OcrNameplatesAlfa::expandWindow(cv::Rect& window, std::vector<Detection> const& dets) {
    assert(isSortedByXMid(dets));

    int vacancy = 17 - int(dets.size());
    if (vacancy <= 0) return window;

    double charWidth = double(computeExtent(dets).width) / dets.size();
    double additionalWidth = charWidth * vacancy * 1.05;

    int newX = window.x;
    int newY = window.y;
    int newW = int(round(window.width + additionalWidth));
    int newH = window.height;

    double slope = computeCharAlignmentSlope(dets);

    if (slope > 0) {
        newH += round(slope * additionalWidth);
    } else {
        newY += round(slope * additionalWidth);
    }

    window.x = newX;
    window.y = newY;
    window.width = newW;
    window.height = newH;

    return window;
}

bool OcrNameplatesAlfa::isWindowTooLarge(cv::Rect const& window, cv::Rect const& detsExtent) {
    return (window.width - detsExtent.width > 2.5 * WINDOW_X_BORDER) || (window.height - detsExtent.height > 2.5 * WINDOW_Y_BORDER);
}

cv::Rect& OcrNameplatesAlfa::adjustWindow(cv::Rect& window, cv::Rect const& detsExtent) {
    int newLeft = window.x + (detsExtent.x - WINDOW_X_BORDER);
    int newRight = window.x + (detsExtent.x + detsExtent.width + WINDOW_X_BORDER);
    int newTop = window.y + (detsExtent.y - WINDOW_Y_BORDER);
    int newBottom = window.y + (detsExtent.y + detsExtent.height + WINDOW_Y_BORDER);

    window.x = newLeft;
    window.y = newTop;
    window.width = newRight - newLeft;
    window.height = newBottom - newTop;

    return window;
}

int OcrNameplatesAlfa::estimateCharSpacing(std::vector<Detection> const& dets) {
    if (dets.size() < 2) return 0;
    assert(isSortedByXMid(dets));

    std::vector<int> spacings;
    for (auto itr = dets.begin() + 1; itr != dets.end(); ++itr) {
        int spacing = OcrUtils::computeSpacing((itr - 1)->getRect(), itr->getRect());
        spacings.push_back(spacing);
    }

    return int(OcrUtils::findMedian(spacings, [](int spacing){ return spacing; }));
}

void OcrNameplatesAlfa::addGapDetections(std::vector<Detection>& dets, cv::Rect const& window) const {
    if (dets.size() <= 2 || dets.size() >= 17) return;
    assert(isSortedByXMid(dets));

    std::vector<Detection> addedDets;

    int spacingRef = estimateCharSpacing(dets);
    for (auto itr = dets.begin() + 1; itr != dets.end(); ++itr) {
        cv::Rect leftRect = (itr - 1)->getRect();
        cv::Rect rightRect = itr->getRect();
        if (OcrUtils::computeSpacing(leftRect, rightRect) > 1.5 * spacingRef) {
            int gapX = leftRect.x + leftRect.width - CHAR_X_BORDER;
            int gapY = (leftRect.y + rightRect.y) / 2 - CHAR_Y_BORDER;
            int gapW = (rightRect.x - (leftRect.x + leftRect.width)) + CHAR_X_BORDER * 2;
            int gapH = (leftRect.height + rightRect.height) / 2 + CHAR_Y_BORDER * 2;

            int gapXReal = window.x + gapX;
            int gapYReal = window.y + gapY;

            cv::Rect gapRect;
            gapRect.x = gapXReal;
            gapRect.y = gapYReal;
            gapRect.width = gapW;
            gapRect.height = gapH;

            OcrUtils::validateWindow(gapRect, _image);
            cv::Mat gapExpanded(gapRect.height, gapRect.width * 10, CV_8UC3, cv::Scalar(0, 0, 0));
            cv::Rect expandedCenter(cv::Point(gapExpanded.cols / 2 - gapRect.width/2, 0), gapRect.size());

            _image(gapRect).copyTo(gapExpanded(expandedCenter));
            std::vector<Detection> gapDets = _pDetectorValues1->detect(gapExpanded);

            if (!gapDets.empty()) {
                gapRect.x = gapX;
                gapRect.y = gapY;
                OcrUtils::validateWindow(gapRect, window);

                Detection& gapDet = gapDets.front();
                gapDet.setRect(gapRect);

                addedDets.push_back(gapDet);
            }
        }
    }

    dets.insert(dets.end(), addedDets.begin(), addedDets.end());
}

void OcrNameplatesAlfa::reexamineCharsWithLowConfidence(std::vector<Detection>& dets, cv::Mat const& roi) const {
    for (auto& det: dets) {
        if (!OcrUtils::isNumbericChar(det.getClass())) continue;
        if (det.getScore() >= 0.8) continue;

        vector<Prediction> pres = _pClassifierChars->Classify(roi(det.getRect()), 1);
        if (pres[0].second > 0.9) {
            det.setClass(pres[0].first);
            det.setScore(pres[0].second);
        }
    }
};

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

// -------- added by WRZ -------- //

DetectedItem OcrNameplatesAlfa::detectValueOfMaxMassAllowed() {
    auto itrKey = _keyDetectedItems.find(CLASSNAME_MAX_MASS_ALLOWED);
    if (itrKey == _keyDetectedItems.end()) return DetectedItem();

    _pDetectorValues2->setThresh(0.05, 0.3);
    cv::Rect keyRect = itrKey->second.rect;

    int x, y, w, h;
    x = keyRect.x + keyRect.width;
    y = keyRect.y + 3;
    w = std::max(int(keyRect.width * 0.1), 95);
    h = std::max(int(keyRect.height * 1), 45);
    cv::Rect window(x, y, std::max(0, w), std::max(0, h));
    OcrUtils::validateWindow(window, _image);

    std::string valueContent;
    int valueLength = 4;
    commonDetectProcess(valueContent, *_pDetectorValues2, _image, window, valueLength);

    return {valueContent, window};
}

DetectedItem OcrNameplatesAlfa::detectValueOfDateOfManufacture() {
    auto itrKey = _keyDetectedItems.find(CLASSNAME_DATE_OF_MANUFACTURE);
    if (itrKey == _keyDetectedItems.end()) return DetectedItem();

    _pDetectorValues2->setThresh(0.05, 0.3);
    cv::Rect keyRect = itrKey->second.rect;

    string result = "";
    int x, y, w, h;
    x = keyRect.x + keyRect.width + 5;
    y = keyRect.y - 2;
    w = std::max(int(keyRect.width * 1.5), 150);
    h = int(keyRect.height * 1.2);
    cv::Rect window(x, y, std::max(0, w), std::max(0, h));

    int valueLength = 6;
    commonDetectProcess(result, *_pDetectorValues2, _image, window, valueLength);
    return {result, window};
}

DetectedItem OcrNameplatesAlfa::detectValueOfMaxNetPowerOfEngine() {
    auto itrKey = _keyDetectedItems.find(CLASSNAME_MAX_NET_POWER_OF_ENGINE);
    if (itrKey == _keyDetectedItems.end()) return DetectedItem();

    _pDetectorValues2->setThresh(0.05, 0.3);
    cv::Rect keyRect = itrKey->second.rect;

    string result = "";
    int x, y, w, h;
    x = keyRect.x + keyRect.width;
    y = keyRect.y + 10;
    w = std::max(int(keyRect.width * 0.1), 90);
    h = std::max(int(keyRect.height * 1), 40);
    cv::Rect window(x, y, std::max(0, w), std::max(0, h));

    int valueLength = 3;

    commonDetectProcess(result, *_pDetectorValues2, _image, window, valueLength);

    if (result.empty()) {
        x = keyRect.x + keyRect.width;
        y = keyRect.y - 5;
        w = std::max(int(keyRect.width * 0.1), 90);
        h = std::max(int(keyRect.height * 1), 40);
        window = cv::Rect(x, y, std::max(0, w), std::max(0, h));
        commonDetectProcess(result, *_pDetectorValues2, _image, window, valueLength);
    }

    for (int i = 0; i < result.size(); ++i) {
        if (result.substr(i, 1) == "5" || result.substr(i, 1) == "3" || result.substr(i, 1) == "6" ||
            result.substr(i, 1) == "2" || result.substr(i, 1) == "0") {
            result = "206";
        } else if (result.substr(i, 1) == "1" || result.substr(i, 1) == "4" || result.substr(i, 1) == "7") {
            result = "147";
        }
    }

    return {result, window};
}

DetectedItem OcrNameplatesAlfa::detectValueOfNumPassengers() {
    auto itrKey = _keyDetectedItems.find(CLASSNAME_NUM_PASSENGERS);
    if (itrKey == _keyDetectedItems.end()) return DetectedItem();

    _pDetectorValues1->setThresh(0.2, 0.1);
    cv::Rect keyRect = itrKey->second.rect;

    string result = "";
    int x, y, w, h;
    x = keyRect.x + keyRect.width + 5;
    y = keyRect.y;
    w = std::max(int(keyRect.width * 0.1), 45);
    h = std::max(int(keyRect.height * 1), 40);
    cv::Rect window(x, y, std::max(0, w), std::max(0, h));

    cv::Mat subImg;
    _image(window).copyTo(subImg);

    vector<Detection> valueDets = _pDetectorValues1->detect(subImg);

    sortByXMid(valueDets);

    eliminateOverlaps(valueDets);

    vector<Detection> temp1;
    for (auto const& det: valueDets) {
        string cls = det.getClass();
        if (OcrUtils::isNumbericChar(cls)) temp1.push_back(det);
    }

    vector<Detection> temp2;
    if (temp1.size() > 1) {
        sortByScoreDescending(temp1);
        temp2.insert(temp2.end(), temp1.begin(), temp1.begin() + 1);
        sortByXMid(temp2);
        for (auto const& det: temp2) {
            result += det.getClass();
        }
    } else {
        temp2 = temp1;
        for (auto const& det: temp2) {
            result += det.getClass();
        }
    }

    return {result, window};
}

DetectedItem OcrNameplatesAlfa::detectValueOfEngineModel() {
    auto itrKey = _keyDetectedItems.find(CLASSNAME_ENGINE_MODEL);
    if (itrKey == _keyDetectedItems.end()) return DetectedItem();

    _pDetectorValues2->setThresh(0.05, 0.3);
    cv::Rect keyRect = itrKey->second.rect;

    string result = "";
    int x, y, w, h;
    x = keyRect.x + keyRect.width + 5;
    y = keyRect.y;
    w = std::max(int(keyRect.width * 0.1), 150);
    h = std::max(int(keyRect.height * 1), 40);
    cv::Rect window(x, y, std::max(0, w), std::max(0, h));

    int valueLength = 8;
    commonDetectProcess(result, *_pDetectorValues2, *_pClassifierChars, _image, window, valueLength);

    return {result, window};
}

DetectedItem OcrNameplatesAlfa::detectValueOfVehicleModel() {
    auto itrKey = _keyDetectedItems.find(CLASSNAME_VEHICLE_MODEL);
    if (itrKey == _keyDetectedItems.end()) return DetectedItem();

    _pDetectorValues2->setThresh(0.05, 0.3);
    cv::Rect keyRect = itrKey->second.rect;

    string result = "";
    int x, y, w, h;
    x = std::min(keyRect.x + keyRect.width + 5, _image.cols - 1);
    y = keyRect.y;
    w = std::max(std::min(int(_image.cols - 1 - x), 120), 0);
    h = std::max(int(keyRect.height * 1), 40);
    cv::Rect window(x, y, std::max(0, w), std::max(0, h));

    int valueLength = 8;
    bool containEnglishChar = true;
    commonDetectProcessForVehicleModel(result, *_pDetectorValues2, _image, window, valueLength, containEnglishChar);

    return {result, window};
}

DetectedItem OcrNameplatesAlfa::detectValueOfEngineDisplacement() {
    auto itrKey = _keyDetectedItems.find(CLASSNAME_ENGINE_DISPLACEMENT);
    if (itrKey == _keyDetectedItems.end()) return DetectedItem();

    _pDetectorValues2->setThresh(0.05, 0.3);
    cv::Rect keyRect = itrKey->second.rect;

    string result = "";
    int x, y, w, h;
    x = keyRect.x + keyRect.width + 5;
    y = keyRect.y;//keyRect.y-5;
    w = std::max(int(keyRect.width * 0.1), 100);
    h = std::max(int(keyRect.height * 1), 35);
    cv::Rect window(x, y, std::max(0, w), std::max(0, h));

    int valueLength = 4;
    commonDetectProcess(result, *_pDetectorValues2, _image, window, valueLength);
    return {result, window};
}

std::string OcrNameplatesAlfa::matchPaint(std::string const& str1, std::string const& str2) {
    std::vector<std::string> names = {"414", "361", "217", "248", "092", "093", "035", "318", "408", "409", "620"};
    if (str1 == "0" && str2 == "2")
        return "092";
    if (str1 == "4" && str2 == "4")
        return "414";

    string result = "";
    int flag = -1;

    for (int i = 0; i < 11; ++i) {
        string s1, s2, s3;
        s1 = names[i].substr(0, 1);
        s2 = names[i].substr(1, 1);
        s3 = names[i].substr(2, 1);
        if ((str1 == s1 && str2 == s2) || (str1 == s2 && str2 == s3)) {
            flag = i;
        }
    }

    if (flag > -1) {
        result = names[flag];
    }

    return result;
}

DetectedItem OcrNameplatesAlfa::detectValueOfPaint() {
    auto itrKey = _keyDetectedItems.find(CLASSNAME_PAINT);
    if (itrKey == _keyDetectedItems.end()) return DetectedItem();

    _pDetectorValues2->setThresh(0.05, 0.3);
    cv::Rect keyRect = itrKey->second.rect;

    string result = "";
    int x, y, w, h;
    x = keyRect.x + keyRect.width;
    y = keyRect.y;
    w = std::max(int(keyRect.width * 0.1), 45);
    h = std::max(int(keyRect.height * 1), 30);
    cv::Rect window(x, y, std::max(0, w), std::max(0, h));

    int valueLength = 3;
    commonDetectProcess(result, *_pDetectorValues2, _image, window, valueLength); // thresh = 0.05?

    if (result.size() == 2) {
        string s1 = result.substr(0, 1);
        string s2 = result.substr(1, 1);
        result = matchPaint(s1, s2);
    } else if (result.size() == 3) {
        string s1 = result.substr(0, 1);
        string s2 = result.substr(1, 1);
        string resultMatch = matchPaint(s1, s2);
        if (resultMatch.empty()) {
            s1 = result.substr(1, 1);
            s2 = result.substr(2, 1);
            resultMatch = matchPaint(s1, s2);
        }
        result = resultMatch;
    }

    return {result, window};
}

void OcrNameplatesAlfa::commonDetectProcess(string& result, PvaDetector& detectorValues, cv::Mat const& img,
                                            cv::Rect const& window, int valueLength, bool containsLetters,
                                            float confThresh, float iouThresh) {
    result = "";

    vector<Detection> temp1;
    subCommonDetectProcess(detectorValues, img, window, temp1, containsLetters, confThresh, iouThresh);

    if (temp1.empty()) return;


    sortByXMid(temp1);
    vector<Detection> temp2;
    bool isMoved = false;
    cv::Rect windowMoved;

    if (temp1.size() == valueLength) {
        temp2 = temp1;
        for (auto const& det: temp2) {
            if (det.getClass() == "Z") result += "7";
            else result += det.getClass();
        }
    } else {
        isMoved = moveWindow(img, temp1, window, windowMoved);

        subCommonDetectProcess(detectorValues, img, windowMoved, temp2, containsLetters, confThresh, iouThresh);

        if (temp2.size() > valueLength) {
            sortByScoreDescending(temp2);
            vector<Detection> temp3;
            temp3.insert(temp3.end(), temp2.begin(), temp2.begin() + valueLength);
            sortByXMid(temp3);
            for (auto const& det: temp3) {
                if (det.getClass() == "Z") result += "7";
                else result += det.getClass();
            }
            temp2.clear();
            temp2.insert(temp2.end(), temp3.begin(), temp3.end());
        }
        else if (temp2.size() <= valueLength) {
            sortByXMid(temp2);
            for (auto const& det: temp2) {
                if (det.getClass() == "Z") result += "7";
                else result += det.getClass();
            }
        }
    }
}

void OcrNameplatesAlfa::commonDetectProcess(string& result, PvaDetector& detectorValues, Classifier& classifier,
                                            cv::Mat const& img, cv::Rect const& window, int valueLength,
                                            bool containEnglish, float conf, float iouThresh) {
    result = "";
    vector<Detection> temp1;
    subCommonDetectProcess(detectorValues, img, window, temp1, containEnglish, conf, iouThresh);

    if (temp1.empty()) return;

    sortByXMid(temp1);
    vector<Detection> temp2;
    bool isMoved = false;
    cv::Rect windowMoved;

    if (temp1.size() == valueLength) {
        temp2 = temp1;
        for (auto const& det: temp2) {
            if (det.getClass() == "Z") result += "7";
            else result += det.getClass();
        }
    } else {
        isMoved = moveWindow(img, temp1, window, windowMoved);

        subCommonDetectProcess(detectorValues, img, windowMoved, temp2, containEnglish, conf, iouThresh);

        if (temp2.size() > valueLength) {
            sortByScoreDescending(temp2);
            vector<Detection> temp3;
            temp3.insert(temp3.end(), temp2.begin(), temp2.begin() + valueLength);
            temp2.clear();
            temp2.insert(temp2.end(), temp3.begin(), temp3.end());
        }
    }


    result = "";

    cv::Mat subImg;
    img(window).copyTo(subImg);
    cropImg(subImg);
    if (isMoved) {
        subImg = img(windowMoved).clone();
        cropImg(subImg);
    }

    sortByXMid(temp2);
    for (auto const& det: temp2) {
        cv::Mat cropImg = subImg(det.getRect()).clone();
        cv::resize(cropImg, cropImg, cv::Size(224, 224));
        vector<Prediction> pre = classifier.Classify(cropImg, 1);
        if (det.getScore() < 0.99) {
            if (pre[0].second < 0.85 || !OcrUtils::isNumbericChar(pre[0].first)) {
                result += det.getClass();
            } else {
                result += pre[0].first;
            }
        } else {
            result += det.getClass();
        }
    }
}

void OcrNameplatesAlfa::commonDetectProcessForVehicleModel(string& result, PvaDetector& detectorValues,
                                                      cv::Mat const& img, cv::Rect const& window, int valueLength,
                                                      bool containsLetters, float conf, float iouThresh) {
    result = "";
    vector<Detection> temp1;
    subCommonDetectProcess(detectorValues, img, window, temp1, containsLetters, conf, iouThresh);

    if (temp1.empty()) return;

    sortByXMid(temp1);
    eliminateOverlaps(temp1);

    vector<Detection> temp2;
    bool isMoved = false;
    cv::Rect windowMoved;

    if (temp1.size() == valueLength) {
        temp2 = temp1;
        for (auto const& det: temp2) {
            if (det.getClass() == "Z") result += "7";
            else result += det.getClass();
        }
    } else {
        isMoved = moveWindow(img, temp1, window, windowMoved);

        subCommonDetectProcess(detectorValues, img, windowMoved, temp2, containsLetters, conf, iouThresh);

        if (temp2.size() > valueLength) {
            sortByScoreDescending(temp2);
            vector<Detection> temp3;
            temp3.insert(temp3.end(), temp2.begin(), temp2.begin() + valueLength);
            sortByXMid(temp3);
            for (int i = 0; i < valueLength; ++i) {
                if (temp3[i].getClass() == "Z") result += "7";
                else result += temp3[i].getClass();
            }
            temp2.clear();
            temp2.insert(temp2.end(), temp3.begin(), temp3.end());
        }

        else if (temp2.size() <= valueLength) {
            sortByXMid(temp2);
            for (auto const& det: temp2) {
                if (det.getClass() == "Z") result += "7";
                else result += det.getClass();
            }
        }
    }
}

void OcrNameplatesAlfa::subCommonDetectProcess(PvaDetector& detectorValues, cv::Mat const& img, cv::Rect const& window, vector<Detection> &dets,
                                               bool containsLetters, float conf, float iouThresh) {
    cv::Mat subimg;
    img(window).copyTo(subimg);
    cropImg(subimg);

    detectorValues.setThresh(conf,iouThresh);
    vector<Detection> valueDets = detectorValues.detect(subimg);

    vector<Detection> temp;
    if (!containsLetters){
        for(Detection const& det: valueDets){
            string cls = det.getClass();
            if (OcrUtils::isNumbericChar(cls)) temp.push_back(det);
        }
    } else{
        temp = valueDets;
    }

    sortByXMid(temp);
    eliminateYOutliers(temp);
    eliminateOverlaps(temp);

    dets.clear();
    dets.insert(dets.end(), temp.begin(), temp.end());
}

bool OcrNameplatesAlfa::moveWindow(cv::Mat const& img, vector<Detection>& dets, cv::Rect const& window, cv::Rect& newWindow) {
    int threshy = 100;
    bool needmove = false;
    newWindow = window;
    if (dets.empty()) return false;
    int cy = 0;
    for (Detection const& det:dets) {
        if (det.getRect().y < threshy || abs(det.getRect().y + det.getRect().height - 640) < threshy) {
            needmove = true;
            cy = det.getRect().y + det.getRect().height / 2;
        }
    }
    if (needmove) {
        float ratio;
        if (float(window.width) / window.height > 1056.0 / 640.0) ratio = float(1056) / window.width;
        else ratio = float(640) / window.height;
        newWindow.y = window.y + int((cy - 320) / (ratio));
    }
    return needmove;
}

void OcrNameplatesAlfa::cropImg(cv::Mat& input) {
    cv::Mat subImg(642, 1058, CV_8UC3, cv::Scalar(0, 0, 0));
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
        cv::Rect window(528 - nw / 2, 0, nw, nh);
        input.copyTo(subImg(window));
    } else {
        // w is main
        ratio = float(1056) / w;
        nh = int(h * ratio);
        nw = 1056;
        cv::resize(input, input, cv::Size(nw, nh));
        cv::Rect window(0, std::max(0, 320 - nh / 2), nw, nh);
        input.copyTo(subImg(window));
    }
    subImg.copyTo(input);
}