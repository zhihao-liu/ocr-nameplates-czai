//
// Created by Zhihao Liu on 18-4-4.
//

#ifndef OCR_CUIZHOU_OCRNAMEPLATES_H
#define OCR_CUIZHOU_OCRNAMEPLATES_H

#include <map>
#include "ocr_implementation/ocr_handler.h"
#include "ocr_aux/keyvalue_detection.h"
#include "ocr_aux/classname_dict.hpp"


namespace cz {

class OcrNameplate : public OcrHandler {
protected:
    enum class NameplateField { UNKNOWN = -1, VIN = 0, MANUFACTURER, BRAND, MAX_MASS_ALLOWED, MAX_NET_POWER_OF_ENGINE, COUNTRY, FACTORY, ENGINE_MODEL, NUM_PASSENGERS, VEHICLE_MODEL, ENGINE_DISPLACEMENT, DATE_OF_MANUFACTURE, PAINT };

public:
    virtual ~OcrNameplate() override;

    virtual cv::Mat drawResult() const override;

    virtual std::string getResultAsString() const override;
    std::vector<KeyValueDetection> getResultAsArray() const;

//    void printResultToConsoleInChinese() const;
//    void printResultToFileInChinese(std::ofstream& outFile) const;

protected:
    static ClassnameDict<NameplateField> const fieldDict_;
    std::map<NameplateField, KeyValueDetection> result_;

    OcrNameplate();
};

}


#endif //OCR_CUIZHOU_OCRNAMEPLATES_H
