//
// Created by Zhihao Liu on 18-4-4.
//

#ifndef OCR_CUIZHOU_OCRNAMEPLATESALFA_H
#define OCR_CUIZHOU_OCRNAMEPLATESALFA_H

#include "ocr_nameplates.h"
#include "pvadetector.h"
#include "classifier.h"


namespace cuizhou {
    class OcrNameplatesAlfa final : public OcrNameplates {
    public:
        ~OcrNameplatesAlfa() override = default;
        OcrNameplatesAlfa() = default;
        OcrNameplatesAlfa(PvaDetector& detectorKeys, PvaDetector& detectorValues1, PvaDetector& detectorValues2, Classifier& classifierChars);

        virtual void processImage() override;

    private:
        static int const WINDOW_X_BORDER;
        static int const WINDOW_Y_BORDER;
        static int const CHAR_X_BORDER;
        static int const CHAR_Y_BORDER;
        static std::vector<std::string> const PAINT_CANDIDATES;

        PvaDetector* pDetectorKeys_;
        PvaDetector* pDetectorValuesVin_; // used for VIN and NumPassengers
        PvaDetector* pDetectorValuesOther_; // used for other values
        Classifier* pClassifierChars_;
        std::map<std::string, DetectedItem> keyDetectedItems_;

        void detectKeys();
        void adaptiveRotationWithUpdatingKeyDetections();
        DetectedItem detectValue(std::string const& keyName);
        DetectedItem detectValueOfVin();

        static cv::Rect estimateValueRectOfVin(cv::Rect const& keyRect);
        static cv::Rect estimateValueRectOf;

        static std::string joinDetectedChars(std::vector<Detection> const& dets);

        void addGapDetections(std::vector<Detection>& dets, cv::Rect const& window) const;
        void reexamineCharsWithLowConfidence(std::vector<Detection>& dets, cv::Mat const& roi) const;
        static void mergeOverlappedDetections(std::vector<Detection>& dets);

        static void sortByXMid(std::vector<Detection>& dets);
        static void sortByYMid(std::vector<Detection>& dets);
        static void sortByScoreDescending(std::vector<Detection>& dets);
        static bool isSortedByXMid(std::vector<Detection> const& dets);

        static bool containsUnambiguousNumberOne(Detection const& det1, Detection const& det2);
        static void eliminateYOutliers(std::vector<Detection>& dets);
        static void eliminateOverlaps(std::vector<Detection>& dets);
        static cv::Rect computeExtent(std::vector<Detection> const& dets);
        static double computeCharAlignmentSlope(std::vector<Detection> const& dets);
        static int estimateCharSpacing(std::vector<Detection> const& dets);

        static cv::Rect expandWindow(cv::Rect const& roi, std::vector<Detection> const& dets);
        static bool isWindowTooLarge(cv::Rect const& roi, cv::Rect const& detsExtent);
        static cv::Rect adjustWindow(cv::Rect const& roi, cv::Rect const& detsExtent);

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

        static void commonDetectProcess(string& result, PvaDetector& detectorValues, cv::Mat const& img, cv::Rect const& window, int valueLength, bool containsLetters = false, float conf = 0.2, float iouThresh = 0.1);
        static void commonDetectProcess(string& result, PvaDetector& detectorValues, Classifier& classifier, cv::Mat const& img, cv::Rect const& window, int valueLength, bool containsLetters = false, float conf = 0.2, float iouThresh=0.1);
        static void commonDetectProcessForVehicleModel(string& result, PvaDetector& detectorValues, cv::Mat const& img, cv::Rect const& window, int valueLength, bool containsLetters = false, float conf = 0.2, float iouThresh = 0.1);
        static void subCommonDetectProcess(PvaDetector& detectorValues, cv::Mat const& img, cv::Rect const& window, vector<Detection>& dets, bool containsLetters = false, float conf = 0.2, float iouThresh = 0.1);
        static bool moveWindow(cv::Mat const& img, vector<Detection>& dets, cv::Rect const& window, cv::Rect& newWindow);
        static void cropImg(cv::Mat& input);
    };
}


#endif //OCR_CUIZHOU_OCRNAMEPLATESALFA_H
