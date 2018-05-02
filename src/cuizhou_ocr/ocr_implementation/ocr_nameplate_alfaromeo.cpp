//
// Created by Zhihao Liu on 18-4-4.
//

#include "ocr_implementation/ocr_nameplate_alfaromeo.h"
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include "data_utils/cv_extension.h"
#include "data_utils/data_proc.hpp"
#include "ocr_aux/detection_proc.h"


namespace cz {

//std::vector<std::string> const OcrNameplateAlfaRomeoromeo::PAINT_CANDIDATES = {"414", "361", "217", "248", "092", "093", "035", "318", "408", "409", "620"};

EnumHashMap<OcrNameplateAlfaRomeo::NameplateField, int> const OcrNameplateAlfaRomeo::VALUE_LENGTH = {
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

/* ------- Auxiliary Functions Hidden in Anonymous Namespace ------- */
namespace {

int const STANDARD_IMG_WIDTH = 1024;
int const STANDARD_IMG_HEIGHT = 768;
int const ROI_X_BORDER = 8;
int const ROI_Y_BORDER = 4;
int const CHAR_X_BORDER = 2;
int const CHAR_Y_BORDER = 1;

cv::Rect& extendRoiCoverage(cv::Rect& roi, std::vector<Detection> const& dets) {
    assert(isSortedByXMid(dets));

    int charSpacing = estimateCharSpacing(dets);
    roi.width = static_cast<int>(charSpacing * 17 * 1.1);

    int vacancy = 17 - static_cast<int>(dets.size());
    if (vacancy > 0) {
        double slope = estimateCharAlignmentSlope(dets);
        (slope > 0 ? roi.height : roi.y) += std::round(slope * charSpacing * vacancy);
    }

    return roi;
}

bool isRoiTooLargeForDetsExtent(cv::Rect const& roi, cv::Rect const& detsExtentInRoi) {
    return isRectTooLarge(roi, detsExtentInRoi,
                          static_cast<int>(2.5 * ROI_X_BORDER),
                          static_cast<int>(2.5 * ROI_Y_BORDER));
}

cv::Rect& adjustRoiToDetsExtent(cv::Rect& roi, cv::Rect detsExtentInRoi) {
    expandRect(detsExtentInRoi, ROI_X_BORDER, ROI_Y_BORDER);
    shrinkRectToExtent(roi, detsExtentInRoi);
    return roi;
}

void resolveOverlappedDetections(std::vector<Detection>& dets) {
    if (dets.size() <= 17) return;

    std::vector<float> overlaps;
    for (auto itr = std::next(dets.cbegin()); itr != dets.cend(); ++itr) {
        float iou = computeIou(std::prev(itr)->rect, itr->rect);
        overlaps.push_back(iou);
    }

    auto itrMaxOverlap = std::max_element(overlaps.cbegin(), overlaps.cend());
    int idxMaxOverlap = static_cast<int>(std::distance(overlaps.cbegin(), itrMaxOverlap));

    int idxToErase = dets[idxMaxOverlap].score < dets[idxMaxOverlap + 1].score ? idxMaxOverlap : idxMaxOverlap + 1;
    dets.erase(std::next(dets.begin(), idxToErase));

    resolveOverlappedDetections(dets);
}

bool containsUnambiguousNumberOne(Detection const& lhs, Detection const& rhs) {
    return (lhs.label == "1" && rhs.label != "7" && rhs.label != "4" && rhs.label != "H")
           || (rhs.label == "1" && lhs.label != "7" && lhs.label != "4" && lhs.label != "H");
}

void eliminateOverlapsByThresh(std::vector<Detection>& dets,
                               std::pair<float, float> firstThresh,
                               std::pair<float, float> secondThresh,
                               float lowConfThresh) {
    if (dets.size() < 2) return;
    assert(isSortedByXMid(dets));

    auto isOverlapped = [&](Detection const& det1, Detection const& det2) {
        float firstTol = containsUnambiguousNumberOne(det1, det2) ? firstThresh.first : firstThresh.second;
        float secondTol = containsUnambiguousNumberOne(det1, det2) ? secondThresh.first : secondThresh.second;
        float overlap = computeIou(det1.rect, det2.rect);
        return overlap > firstTol ||
               (overlap > secondTol && std::min(det1.score, det2.score) < lowConfThresh);
    };
    auto compareScore = [](Detection const& lhs, Detection const& rhs) {
        return lhs.score < rhs.score;
    };

    resolveConflicts(dets, isOverlapped, compareScore);
}

}
/* ------- End Auxiliary Functions ------- */

OcrNameplateAlfaRomeo::~OcrNameplateAlfaRomeo() = default;

OcrNameplateAlfaRomeo::OcrNameplateAlfaRomeo(Detector detectorKeys,
                                             Detector detectorValuesVin,
                                             Detector detectorValuesStitched,
                                             Classifier classifierChars)
        : detectorKeys_(std::move(detectorKeys)),
          detectorValuesVin_(std::move(detectorValuesVin)),
          detectorValuesStitched_(std::move(detectorValuesStitched)),
          classifierChars_(std::move(classifierChars)) {}

void OcrNameplateAlfaRomeo::processImage(ShowProgress const& showProgress) {
    result_.clear();

    image_ = imgResizeAndFill(image_, STANDARD_IMG_WIDTH, STANDARD_IMG_HEIGHT);

    detectKeys();
    adaptiveRotationWithUpdatingKeyDetections();

    detectValueOfVin();
    detectValuesOfOtherCodeFields();
}

void OcrNameplateAlfaRomeo::detectKeys() {
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

void OcrNameplateAlfaRomeo::adaptiveRotationWithUpdatingKeyDetections() {
    auto itrKeyVin = keyOcrDetections_.find(NameplateField::VIN);
    if (itrKeyVin == keyOcrDetections_.end()) return;

    detectorValuesVin_.setThresh(0.1, 0.3);

    cv::Rect const& keyRoi = itrKeyVin->second.rect;
    cv::Rect valueRoi = estimateValueRoi(NameplateField::VIN, keyRoi);
    std::vector<Detection> valueDets = detectorValuesVin_.detect(image_(valueRoi));

    sortByXMid(valueDets);
    eliminateOverlaps(valueDets, NameplateField::VIN);
    eliminateYOutliers(valueDets);

    double slope = estimateCharAlignmentSlope(valueDets);
    if (std::abs(slope) > 0.025) {
        double angle = std::atan(slope) / CV_PI * 180;
        image_ = imgRotate(image_, angle);
        detectKeys(); // update detections of keys
    }
}

void OcrNameplateAlfaRomeo::detectValueOfVin() {
    auto itrKeyItemVin = keyOcrDetections_.find(NameplateField::VIN);
    if (itrKeyItemVin == keyOcrDetections_.end()) return;
    OcrDetection const& keyItem = itrKeyItemVin->second;

    detectorValuesVin_.setThresh(0.05, 0.3);

    cv::Rect keyRoi = keyItem.rect;
    cv::Rect valueRoi = estimateValueRoi(NameplateField::VIN, keyRoi);
    valueRoi &= extent(image_);
    // no need to resize and fill because the model for VIN is trained with stretched images
    std::vector<Detection> valueDets = detectorValuesVin_.detect(image_(valueRoi));

    sortByXMid(valueDets);
    eliminateOverlaps(valueDets, NameplateField::VIN);
    eliminateYOutliers(valueDets);

    // second round in the network
    adjustRoiToDetsExtent(valueRoi, computeExtent(valueDets));
    extendRoiCoverage(valueRoi, valueDets);
    valueRoi &= extent(image_);
    valueDets = detectorValuesVin_.detect(image_(valueRoi));

    sortByXMid(valueDets);
    eliminateOverlaps(valueDets, NameplateField::VIN);
    eliminateYOutliers(valueDets);

    cv::Rect detsExtent = computeExtent(valueDets);
    if (isRoiTooLargeForDetsExtent(valueRoi, detsExtent)) {
        // third round in the network
        adjustRoiToDetsExtent(valueRoi, detsExtent);
        valueRoi &= extent(image_);
        valueDets = detectorValuesVin_.detect(image_(valueRoi));

        sortByXMid(valueDets);
        eliminateOverlaps(valueDets, NameplateField::VIN);
        eliminateYOutliers(valueDets);
    }

    if (valueDets.size() < 17) {
        // fourth round in the network
        addGapDetections(valueDets, valueRoi);
        sortByXMid(valueDets);
    }

    if (valueDets.size() > 17) {
        resolveOverlappedDetections(valueDets);
    }

    updateByClassification(valueDets, image_(valueRoi));

    OcrDetection valueItem = joinDetections(valueDets);
    valueItem.rect = PerspectiveTransform(1, valueRoi.x, valueRoi.y).apply(valueItem.rect);

    result_.emplace(NameplateField::VIN, KeyValueDetection(keyItem, valueItem));
}

cv::Rect OcrNameplateAlfaRomeo::estimateValueRoi(NameplateField field, cv::Rect const& keyRoi) {
    switch (field) {
        case NameplateField::VIN: {
            return cv::Rect(keyRoi.br().x + 5,
                            keyRoi.y - static_cast<int>(std::round(keyRoi.height * 0.25)),
                            static_cast<int>(std::round(keyRoi.width * 1.75)),
                            static_cast<int>(std::round(keyRoi.height * 1.5)));
        }

        case NameplateField::MAX_MASS_ALLOWED: {
            return cv::Rect(keyRoi.br().x,
                            keyRoi.y - static_cast<int>(std::round(keyRoi.height * -0.1)),
                            std::max(static_cast<int>(keyRoi.width * 0.7), 100),
                            std::min(static_cast<int>(keyRoi.height * 1.25), 50));
        }

        case NameplateField::MAX_NET_POWER_OF_ENGINE: {
            return cv::Rect(keyRoi.br().x + 5,
                            keyRoi.y - static_cast<int>(std::round(keyRoi.height * -0.1)),
                            std::max(static_cast<int>(keyRoi.width * 0.5), 95),
                            std::min(static_cast<int>(keyRoi.height * 1.25), 55));
        }

        case NameplateField::ENGINE_MODEL: {
            return cv::Rect(keyRoi.br().x + 10,
                            keyRoi.y - static_cast<int>(std::round(keyRoi.height * 0.1)),
                            std::max(static_cast<int>(keyRoi.width * 1.0), 150),
                            std::min(static_cast<int>(keyRoi.height * 1.25), 40));
        }

        case NameplateField::NUM_PASSENGERS: {
            return cv::Rect(keyRoi.br().x + 5,
                            keyRoi.y - static_cast<int>(std::round(keyRoi.height * 0.05)),
                            std::max(static_cast<int>(keyRoi.width * 0.4), 40),
                            std::min(static_cast<int>(keyRoi.height * 1.15), 40));
        }

        case NameplateField::VEHICLE_MODEL: {
            return cv::Rect(keyRoi.br().x + 5,
                            keyRoi.y - static_cast<int>(std::round(keyRoi.height * 0.1)),
                            std::max(static_cast<int>(keyRoi.width * 1.2), 120),
                            std::min(static_cast<int>(keyRoi.height * 1.25), 40));
        }

        case NameplateField::ENGINE_DISPLACEMENT: {
            return cv::Rect(keyRoi.br().x + 10,
                            keyRoi.y - static_cast<int>(std::round(keyRoi.height * 0.1)),
                            std::max(static_cast<int>(keyRoi.width * 0.8), 80),
                            std::min(static_cast<int>(keyRoi.height * 1.25), 40));
        }

        case NameplateField::DATE_OF_MANUFACTURE: {
            return cv::Rect(keyRoi.br().x + 10,
                            keyRoi.y - static_cast<int>(std::round(keyRoi.height * 0.1)),
                            std::max(static_cast<int>(keyRoi.width * 1.0), 75),
                            std::min(static_cast<int>(keyRoi.height * 1.25), 40));
        }

        case NameplateField::PAINT: {
            return cv::Rect(keyRoi.br().x + 5,
                            keyRoi.y - static_cast<int>(std::round(keyRoi.height * 0.1)),
                            std::max(static_cast<int>(keyRoi.width * 1.2), 45),
                            std::min(static_cast<int>(keyRoi.height * 1.25), 35));
        }

        default: {
            return cv::Rect();
        }
    }
}

void OcrNameplateAlfaRomeo::eliminateOverlaps(std::vector<Detection>& dets, NameplateField field){
    std::pair<float, float> firstThresh, secondThresh;
    switch (field) {
        // for fields of which value characters are compact
        case NameplateField::VIN: {
            firstThresh = std::make_pair(0.6, 0.4);
            secondThresh = std::make_pair(0.6, 0.3);
        }
        case NameplateField::PAINT: {
            firstThresh = std::make_pair(0.6, 0.5);
            secondThresh = std::make_pair(0.6, 0.4);
        } break;

        default: {
            firstThresh = std::make_pair(0.4, 0.3);
            secondThresh = std::make_pair(0.4, 0.2);
        } break;
    }
    eliminateOverlapsByThresh(dets, firstThresh, secondThresh, 0.2);
}

void OcrNameplateAlfaRomeo::addGapDetections(std::vector<Detection>& dets, cv::Rect const& roi) {
    if (dets.size() <= 2 || dets.size() >= 17) return;
    assert(isSortedByXMid(dets));

    detectorValuesVin_.setThresh(0.05, 0.3);
    std::vector<Detection> addedDets;

    int spacingRef = estimateCharSpacing(dets);
    for (auto itr = std::next(dets.cbegin()); itr != dets.cend(); ++itr) {
        cv::Rect const& leftRect = std::prev(itr)->rect;
        cv::Rect const& rightRect = itr->rect;

        if (computeSpacing(leftRect, rightRect) > 1.5 * spacingRef) {
            int gapX = leftRect.br().x;
            int gapY = (leftRect.y + rightRect.y) / 2;
            int gapW = (rightRect.x - leftRect.br().x);
            int gapH = (leftRect.height + rightRect.height) / 2;
            cv::Rect gapRect(gapX, gapY, gapW, gapH);
            expandRect(gapRect, CHAR_X_BORDER, CHAR_Y_BORDER);

            cv::Rect gapRectReal = gapRect + roi.tl();
            gapRectReal &= extent(image_);

            cv::Size sizeExpanded(gapRectReal.width * 10, gapRectReal.height);
            cv::Mat gapExpanded = imgResizeAndFill(image_(gapRectReal), sizeExpanded);
            std::vector<Detection> gapDets = detectorValuesVin_.detect(gapExpanded);

            if (!gapDets.empty()) {
                Detection& gapDet = gapDets.front();
                gapDet.rect = gapRect & roi;

                addedDets.push_back(std::move(gapDet));
            }
        }
    }

    dets.insert(dets.end(), addedDets.cbegin(), addedDets.cend());
}

void OcrNameplateAlfaRomeo::updateByClassification(std::vector<Detection>& dets, cv::Mat const& srcImg) const {
    for (auto& det : dets) {
        if (!isNumbericChar(det.label)) continue;
        if (det.score >= 0.8) continue;

        if (det.rect.area() == 0) continue;
        std::vector<Classification> clss = classifierChars_.classify(srcImg(det.rect), 1);
        if (clss.front().score > 0.9) {
            det.label = clss.front().label;
            det.score = clss.front().score;
        }
    }
};

void OcrNameplateAlfaRomeo::detectValuesOfOtherCodeFields() {
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
        originRoi &= extent(image_);
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
        // coordinates in dectections are relative to the source image
        // convert them to relative to the roi
        detsExtent = PerspectiveTransform(1, -originRoi.x, -originRoi.y).apply(detsExtent);
        adjustRoiToDetsExtent(originRoi, detsExtent);
        originRoi &= extent(image_);
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
        OcrDetection valueItem = joinDetections(dets);

        result_.emplace(field, KeyValueDetection(keyItem, valueItem));
    }
}

void OcrNameplateAlfaRomeo::postprocessStitchedDetections(EnumHashMap<NameplateField, std::vector<Detection>>& stitchedDets) {
    for (auto& splitItem : stitchedDets) {
        NameplateField field = splitItem.first;
        std::vector<Detection>& dets = splitItem.second;

        sortByXMid(dets);
        eliminateOverlaps(dets, field);
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

bool OcrNameplateAlfaRomeo::shouldContainLetters(NameplateField field) {
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

} // end namespace cz