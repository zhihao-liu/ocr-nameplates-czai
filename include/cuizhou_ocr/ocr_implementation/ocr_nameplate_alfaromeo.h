//
// Created by Zhihao Liu on 18-4-4.
//

#ifndef OCR_CUIZHOU_OCRNAMEPLATESALFA_H
#define OCR_CUIZHOU_OCRNAMEPLATESALFA_H

#include "detector.h"
#include "classifier.h"
#include "ocr_implementation/ocr_nameplate.h"
#include "utils/enum_hashmap.hpp"
#include "ocr_aux/collage.hpp"


namespace cuizhou{

class OcrNameplateAlfaRomeo final : public OcrNameplate {
public:
    ~OcrNameplateAlfaRomeo() override;
    OcrNameplateAlfaRomeo() = delete;

    OcrNameplateAlfaRomeo(Detector detectorKeys,
                          Detector detectorValuesVin,
                          Detector detectorValuesStitched,
                          Classifier classifierChars);

    virtual void processImage() override;

private:
    static EnumHashMap<NameplateField, int> const VALUE_LENGTH;

    Detector detectorKeys_;
    Detector detectorValuesVin_; // used for VIN
    Detector detectorValuesStitched_; // used for detect other values in stitched sub-images
    Classifier classifierChars_;

    std::map<NameplateField, OcrDetection> keyOcrDetections_;

    void detectKeys();
    void detectValueOfVin();
    void detectValuesOfOtherCodeFields();
    void adaptiveRotationWithUpdatingKeyDetections();

    void addGapDetections(std::vector<Detection>& dets, cv::Rect const& roi);
    void updateByClassification(std::vector<Detection>& dets, cv::Mat const& srcImg) const;
    static void postprocessStitchedDetections(EnumHashMap<NameplateField, std::vector<Detection>>& stitchedDets);

    static cv::Rect estimateValueRoi(NameplateField field, cv::Rect const& keyRoi);
    static void eliminateOverlaps(std::vector<Detection>& dets, NameplateField field);
    static bool shouldContainLetters(NameplateField field);
};

} // end namespace cuizhou


#endif //OCR_CUIZHOU_OCRNAMEPLATESALFA_H
