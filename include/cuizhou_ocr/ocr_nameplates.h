//
// Created by Zhihao Liu on 18-4-4.
//

#ifndef OCR_CUIZHOU_OCRNAMEPLATES_H
#define OCR_CUIZHOU_OCRNAMEPLATES_H

#include "ocr_handler.h"
#include <map>
#include <unordered_map>
#include <fstream>
#include "key_value_pair.h"
#include "classname_dict.hpp"


namespace cuizhou {
class OcrNameplates : public OcrHandler {
protected:
    enum class NameplateField { UNKNOWN = -1, VIN = 0, MANUFACTURER, BRAND, MAX_MASS_ALLOWED, MAX_NET_POWER_OF_ENGINE, COUNTRY, FACTORY, ENGINE_MODEL, NUM_PASSENGERS, VEHICLE_MODEL, ENGINE_DISPLACEMENT, DATE_OF_MANUFACTURE, PAINT };
    typedef ClassnameDict<NameplateField> NameplateFieldDict;
public:
    virtual ~OcrNameplates() override = default;
    virtual void processImage() override = 0;
    std::vector<KeyValuePair> getResultAsArray() const;
//    void printResultToConsoleInChinese() const;
//    void printResultToFileInChinese(std::ofstream& outFile) const;

protected:
    static NameplateFieldDict const fieldDict_;
    std::map<NameplateField, KeyValuePair> result_;

    OcrNameplates() = default;
};
}


#endif //OCR_CUIZHOU_OCRNAMEPLATES_H
