//
// Created by Zhihao Liu on 18-4-4.
//

#ifndef OCR_CUIZHOU_OCRNAMEPLATESALFA_H
#define OCR_CUIZHOU_OCRNAMEPLATESALFA_H

#include "OcrNameplates.h"
#include "PvaDetector.h"
#include "classifier.h"


namespace cuizhou {
    class OcrNameplatesAlfa final : public OcrNameplates {
    public:
        ~OcrNameplatesAlfa() override;
        OcrNameplatesAlfa();
        OcrNameplatesAlfa(PvaDetector& detectorKeys, PvaDetector& detectorValues, Classifier& classifierChars);

        virtual void processImage() override;

    private:
        static int const WINDOW_X_BORDER;
        static int const WINDOW_Y_BORDER;
        static int const CHAR_X_BORDER;
        static int const CHAR_Y_BORDER;

        PvaDetector* _pDetectorKeys;
        PvaDetector* _pDetectorValues;
        Classifier* _pClassifierChars;
        std::map<std::string, DetectedItem> _keyDetectedItems;

        void detectKeys();
        DetectedItem detectValueOfVin();

        void addGapDetections(std::vector<Detection>& dets, cv::Rect const& window) const;
        void reexamineCharsWithLowConfidence(std::vector<Detection>& dets, cv::Mat const& roi) const;
        static void mergeOverlappedDetections(std::vector<Detection>& dets);

        static std::string joinDetectedChars(std::vector<Detection> const& dets);

        static void sortByXMid(std::vector<Detection>& dets);
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
    };
}


#endif //OCR_CUIZHOU_OCRNAMEPLATESALFA_H
