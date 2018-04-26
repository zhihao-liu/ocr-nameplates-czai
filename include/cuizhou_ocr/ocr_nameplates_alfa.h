//
// Created by Zhihao Liu on 18-4-4.
//

#ifndef OCR_CUIZHOU_OCRNAMEPLATESALFA_H
#define OCR_CUIZHOU_OCRNAMEPLATESALFA_H

#include "ocr_nameplates.h"
#include <unordered_map>
#include "detector.h"
#include "classifier.h"
#include "utils/enum_hashmap.hpp"
#include "data_aux/collage.hpp"


namespace cuizhou {

class OcrNameplatesAlfa final : public OcrNameplates {
public:
    ~OcrNameplatesAlfa() override = default;
    OcrNameplatesAlfa() = default;
    OcrNameplatesAlfa(Detector detectorKeys,
                      Detector detectorValuesVin,
                      Detector detectorValuesStitched,
                      Classifier classifierChars);

    virtual void processImage() override;

private:
    static int const STANDARD_IMG_WIDTH;
    static int const STANDARD_IMG_HEIGHT;
    static int const ROI_X_BORDER;
    static int const ROI_Y_BORDER;
    static int const CHAR_X_BORDER;
    static int const CHAR_Y_BORDER;
    static std::vector<std::string> const PAINT_CANDIDATES;
    static EnumHashMap<NameplateField, int> const VALUE_LENGTH;

    Detector detectorKeys_;
    Detector detectorValuesVin_; // used for VIN
    Detector detectorValuesStitched_; // used for detect other values in stitched sub-images
    Classifier classifierChars_;

    std::map<NameplateField, OcrDetection> keyOcrDetections_;

    void detectKeys();
    void adaptiveRotationWithUpdatingKeyDetections();

    OcrDetection detectValue(NameplateField field);
    void detectValueOfVin();
    void detectValuesOfOtherCodeFields();
    static void postprocessStitchedDetections(EnumHashMap<NameplateField, std::vector<Detection>>& stitchedDets);

    static cv::Rect estimateValueRoi(NameplateField field, cv::Rect const& keyRoi);
    static std::string joinDetectedChars(std::vector<Detection> const& dets);

    void addGapDetections(std::vector<Detection>& dets, cv::Rect const& roi);
    void updateByClassification(std::vector<Detection>& dets, cv::Mat const& srcImg) const;
    static void mergeOverlappedDetections(std::vector<Detection>& dets);

    static void sortByXMid(std::vector<Detection>& dets);
    static void sortByYMid(std::vector<Detection>& dets);
    static void sortByScoreDescending(std::vector<Detection>& dets);
    static bool isSortedByXMid(std::vector<Detection> const& dets);

    static void eliminateLetters(std::vector<Detection>& dets);
    static bool shouldContainLetters(NameplateField field);
    static bool containsUnambiguousNumberOne(Detection const& det1, Detection const& det2);

    static void eliminateXOverlaps(std::vector<Detection>& dets,
                                   std::pair<float, float> firstThresh,
                                   std::pair<float, float> secondThresh,
                                   float lowConfThresh);
    static void eliminateXOverlaps(std::vector<Detection>& dets, NameplateField field);
    static void eliminateYOutliers(std::vector<Detection>& dets);

    static cv::Rect computeExtent(std::vector<Detection> const& dets);
    static double estimateCharAlignmentSlope(std::vector<Detection> const& dets);
    static int estimateCharSpacing(std::vector<Detection> const& dets);

    static cv::Rect& expandRoi(cv::Rect& roi, std::vector<Detection> const& dets);
    static bool isRoiTooLarge(cv::Rect const& roi, cv::Rect const& detsExtent);
    static cv::Rect& adjustRoi(cv::Rect& roi, cv::Rect const& detsExtent);

//    static std::string matchPaint(std::string const& str);
//    static std::string matchPaintWithLengthFixed(std::string const &str);
};

} // end namespace cuizhou


#endif //OCR_CUIZHOU_OCRNAMEPLATESALFA_H
