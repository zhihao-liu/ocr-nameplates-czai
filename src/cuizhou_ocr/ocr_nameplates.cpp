//
// Created by Zhihao Liu on 18-4-4.
//

#include "ocr_nameplates.h"


namespace cuizhou {

OcrNameplates::NameplateFieldDict const OcrNameplates::fieldDict_(
        {NameplateField::VIN, NameplateField::MANUFACTURER, NameplateField::BRAND, NameplateField::MAX_MASS_ALLOWED, NameplateField::MAX_NET_POWER_OF_ENGINE, NameplateField::COUNTRY, NameplateField::FACTORY, NameplateField::ENGINE_MODEL, NameplateField::NUM_PASSENGERS, NameplateField::VEHICLE_MODEL, NameplateField::ENGINE_DISPLACEMENT, NameplateField::DATE_OF_MANUFACTURE, NameplateField::PAINT},
        NameplateField::UNKNOWN,
        {"Vin", "Manufacturer", "Brand", "MaxMassAllowed", "MaxNetPowerOfEngine", "Country", "Factory", "EngineModel", "NumPassengers", "VehicleModel", "EngineDisplacement", "DateOfManufacture", "Paint"},
        std::string(),
        {"车辆识别代号", "制造商", "品牌", "最大允许总质量", "发动机最大净功率", "制造国", "生产厂名", "发动机型号", "乘坐人数", "整车型号", "发动机排量", "制造年月", "涂料"},
        std::string()
);

std::vector<KeyValuePair> OcrNameplates::getResultAsArray() const {
    std::vector<KeyValuePair> resultVector;
    std::transform(result_.cbegin(), result_.cend(), std::back_inserter(resultVector),
                   [](std::pair<NameplateField const, KeyValuePair> const& elem) { return elem.second; });
    return resultVector;
}

//void OcrNameplates::printResultToConsoleInChinese() const {
//    result_.printResultToConsole([](std::string const& key) {
//        auto itrKeyMapped = CLASSNAME_ENG_TO_CHN.find(key);
//        return itrKeyMapped == CLASSNAME_ENG_TO_CHN.end() ?
//               "" : itrKeyMapped->second;
//    });
//}
//
//void OcrNameplates::printResultToFileInChinese(std::ofstream& outFile) const {
//    result_.printResultToFile(outFile, [](std::string const& key) {
//        auto itrKeyMapped = CLASSNAME_ENG_TO_CHN.find(key);
//        return itrKeyMapped == CLASSNAME_ENG_TO_CHN.end() ?
//               "" : itrKeyMapped->second;
//    });
//}

} // end namespace cuizhou
