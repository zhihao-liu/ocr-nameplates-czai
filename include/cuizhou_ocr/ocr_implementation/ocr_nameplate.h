//
// Created by Zhihao Liu on 18-4-4.
//

#ifndef OCR_CUIZHOU_OCRNAMEPLATES_H
#define OCR_CUIZHOU_OCRNAMEPLATES_H

#include <map>
#include <unordered_map>
#include <fstream>
#include "ocr_implementation/ocr_handler.h"
#include "data_aux/keyvalue_detection.h"
#include "data_aux/classname_dict.hpp"


namespace cuizhou {

class OcrNameplates : public OcrHandler {
protected:
    enum class NameplateField { UNKNOWN = -1, VIN = 0, MANUFACTURER, BRAND, MAX_MASS_ALLOWED, MAX_NET_POWER_OF_ENGINE, COUNTRY, FACTORY, ENGINE_MODEL, NUM_PASSENGERS, VEHICLE_MODEL, ENGINE_DISPLACEMENT, DATE_OF_MANUFACTURE, PAINT };
public:
    virtual ~OcrNameplates() override = default;
    virtual void processImage() override = 0;
    virtual cv::Mat drawResult() const override;

    virtual std::string getResultAsString() const override;
    std::vector<KeyValueDetection> getResultAsArray() const;

//    void printResultToConsoleInChinese() const;
//    void printResultToFileInChinese(std::ofstream& outFile) const;

protected:
    static ClassnameDict<NameplateField> const fieldDict_;
    std::map<NameplateField, KeyValueDetection> result_;

    OcrNameplates() = default;
};

}


#endif //OCR_CUIZHOU_OCRNAMEPLATES_H
