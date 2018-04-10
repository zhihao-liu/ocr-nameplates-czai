//
// Created by Zhihao Liu on 18-4-4.
//

#ifndef OCR_CUIZHOU_OCRNAMEPLATESALFA_H
#define OCR_CUIZHOU_OCRNAMEPLATESALFA_H

#include "OcrNameplates.h"
#include "pvadetector.h"
#include "classifier.h"


namespace cuizhou {
    class OcrNameplatesAlfa final : public OcrNameplates {
    public:
        ~OcrNameplatesAlfa() override;
        OcrNameplatesAlfa();
        OcrNameplatesAlfa(PvaDetector& detectorKeys, PvaDetector& detectorValues1, PvaDetector& detectorValues2, Classifier& classifierChars);

        virtual void processImage() override;

    private:
        static int const WINDOW_X_BORDER;
        static int const WINDOW_Y_BORDER;
        static int const CHAR_X_BORDER;
        static int const CHAR_Y_BORDER;

        PvaDetector* _pDetectorKeys;
        PvaDetector* _pDetectorValues1; // used for Vin and NumPassengers
        PvaDetector* _pDetectorValues2; // used for other values
        Classifier* _pClassifierChars;
        std::map<std::string, DetectedItem> _keyDetectedItems;

        void detectKeys();
        DetectedItem detectValue(std::string const& keyName);
        DetectedItem detectValueOfVin();

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

        static cv::Rect& expandWindow(cv::Rect& roi, std::vector<Detection> const& dets);
        static bool isWindowTooLarge(cv::Rect const& roi, cv::Rect const& detsExtent);
        static cv::Rect& adjustWindow(cv::Rect& roi, cv::Rect const& detsExtent);

        static cv::Rect estimateValueRectOfVin(cv::Rect const& keyRect);

        // -------- added by WRZ ------- //
        DetectedItem detectValueOfMaxMassAllowed();
        DetectedItem detectValueOfDateOfManufacture();
        DetectedItem detectValueOfMaxNetPowerOfEngine();
        DetectedItem detectValueOfEngineModel();
        DetectedItem detectValueOfNumPassengers();
        DetectedItem detectValueOfVehicleModel();
        DetectedItem detectValueOfEngineDisplacement();
        DetectedItem detectValueOfPaint();
        static std::string matchPaint(std::string const& str1, std::string const& str2);

        static void commonDetectProcess(string& result, PvaDetector& detectorValues, cv::Mat const& img, cv::Rect const& window, int valueLength, bool containsLetters = false, float conf = 0.2, float iouThresh = 0.1);
        static void commonDetectProcess(string& result, PvaDetector& detectorValues, Classifier& classifier, cv::Mat const& img, cv::Rect const& window, int valueLength, bool containsLetters = false, float conf = 0.2, float iouThresh=0.1);
        static void commonDetectProcessForVehicleModel(string& result, PvaDetector& detectorValues, cv::Mat const& img, cv::Rect const& window, int valueLength, bool containsLetters = false, float conf = 0.2, float iouThresh = 0.1);
        static void subCommonDetectProcess(PvaDetector& detectorValues, cv::Mat const& img, cv::Rect const& window, vector<Detection>& dets, bool containsLetters = false, float conf = 0.2, float iouThresh = 0.1);
        static bool moveWindow(cv::Mat const& img, vector<Detection>& dets, cv::Rect const& window, cv::Rect& newWindow);
        static void cropImg(cv::Mat& input);
    };
}


#endif //OCR_CUIZHOU_OCRNAMEPLATESALFA_H
