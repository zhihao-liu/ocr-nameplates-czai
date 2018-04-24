//
// Created by Zhihao Liu on 18-4-4.
//

#ifndef OCR_CUIZHOU_OCRNAMEPLATESALFA_H
#define OCR_CUIZHOU_OCRNAMEPLATESALFA_H

#include "ocr_nameplates.h"
#include <unordered_map>
#include "detector.h"
#include "classifier.h"
#include "enum_hashmap.hpp"
#include "collage.hpp"


namespace cuizhou {

class OcrNameplatesAlfa final : public OcrNameplates {
public:
    ~OcrNameplatesAlfa() override = default;
    OcrNameplatesAlfa() = default;
    OcrNameplatesAlfa(Detector const& detectorKeys,
                      Detector const& detectorValuesVin,
                      Detector const& detectorValuesOthers,
                      Detector const& detectorValuesStitched,
                      Classifier const& classifierChars);

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
    Detector detectorValuesOthers_; // used for other values
    Detector detectorValuesStitched_; // used for detect other values in stitched sub-images
    Classifier classifierChars_;

    std::map<NameplateField, DetectedItem> keyDetectedItems_;

    void detectKeys();
    void adaptiveRotationWithUpdatingKeyDetections();

    DetectedItem detectValue(NameplateField field);
    DetectedItem detectValueOfVin();
    DetectedItem detectValueOfOthers(NameplateField field);

    static cv::Rect estimateValueRoi(NameplateField field, cv::Rect const& keyRoi);
    Collage<NameplateField> stitchValueSubimgs();

    static std::string joinDetectedChars(std::vector<Detection> const& dets);

    void addGapDetections(std::vector<Detection>& dets, cv::Rect const& roi);
    void reexamineCharsWithLowConfidence(std::vector<Detection>& dets, cv::Mat const& subimg) const;
    static void mergeOverlappedDetections(std::vector<Detection>& dets);

    static void sortByXMid(std::vector<Detection>& dets);
    static void sortByYMid(std::vector<Detection>& dets);
    static void sortByScoreDescending(std::vector<Detection>& dets);
    static bool isSortedByXMid(std::vector<Detection> const& dets);

    static void eliminateLetters(std::vector<Detection>& dets);
    static std::string fallbackToNumber(std::string const& str);
    static bool containsUnambiguousNumberOne(Detection const& det1, Detection const& det2);

    static void eliminateXOverlapsWithThresh(std::vector<Detection>& dets, std::pair<float, float> firstThresh, std::pair<float, float> secondThresh);
    static void eliminateXOverlapsForVin(std::vector<Detection>& dets);
    static void eliminateXOverlapsForOthers(std::vector<Detection>& dets);
    static void eliminateYOutliers(std::vector<Detection>& dets);

    static cv::Rect computeExtent(std::vector<Detection> const& dets);
    static double estimateCharAlignmentSlope(std::vector<Detection> const& dets);
    static int estimateCharSpacing(std::vector<Detection> const& dets);

    static cv::Rect expandRoi(cv::Rect const& roi, std::vector<Detection> const& dets);
    static bool isRoiTooLarge(cv::Rect const& roi, cv::Rect const& detsExtent);
    static cv::Rect adjustRoi(cv::Rect const& roi, cv::Rect const& detsExtent);

    static std::string matchPaint(std::string const& str);
    static std::string matchPaintWithLengthFixed(std::string const &str);

    // -------- added by WRZ ------- //
    DetectedItem detectValueOfMaxMassAllowed();
    DetectedItem detectValueOfDateOfManufacture();
    DetectedItem detectValueOfMaxNetPowerOfEngine();
    DetectedItem detectValueOfEngineModel();
    DetectedItem detectValueOfNumPassengers();
    DetectedItem detectValueOfVehicleModel();
    DetectedItem detectValueOfEngineDisplacement();
    DetectedItem detectValueOfPaint();

    static void commonDetectProcess(string& result, Detector& detectorValues, cv::Mat const& img, cv::Rect const& roi, int valueLength, bool containsLetters = false, float conf = 0.2, float iouThresh = 0.1);
    static void commonDetectProcess(string& result, Detector& detectorValues, Classifier& classifier, cv::Mat const& img, cv::Rect const& roi, int valueLength, bool containsLetters = false, float conf = 0.2, float iouThresh=0.1);
    static void commonDetectProcessForVehicleModel(string& result, Detector& detectorValues, cv::Mat const& img, cv::Rect const& roi, int valueLength, bool containsLetters = false, float conf = 0.2, float iouThresh = 0.1);
    static void subCommonDetectProcess(Detector& detectorValues, cv::Mat const& img, cv::Rect const& roi, vector<Detection>& dets, bool containsLetters = false, float conf = 0.2, float iouThresh = 0.1);
    static bool moveRoi(cv::Mat const& img, vector<Detection>& dets, cv::Rect const& roi, cv::Rect& newRoi);
    static void cropImg(cv::Mat& input);
};

} // end namespace cuizhou


#endif //OCR_CUIZHOU_OCRNAMEPLATESALFA_H
