//
// Created by Zhihao Liu on 18-4-4.
//

#include "OcrNameplates.h"


using namespace cuizhou;

std::string const OcrNameplates::CLASSNAME_VIN = "Vin";
std::string const OcrNameplates::CLASSNAME_MANUFACTURER = "Manufacturer";
std::string const OcrNameplates::CLASSNAME_BRAND = "Brand";
std::string const OcrNameplates::CLASSNAME_MAX_MASS_ALLOWED = "MaxMassAllowed";
std::string const OcrNameplates::CLASSNAME_MAX_NET_POWER_OF_ENGINE = "MaxNetPowerOfEngine";
std::string const OcrNameplates::CLASSNAME_COUNTRY = "Country";
std::string const OcrNameplates::CLASSNAME_FACTORY = "Factory";
std::string const OcrNameplates::CLASSNAME_ENGINE_MODEL = "EngineModel";
std::string const OcrNameplates::CLASSNAME_NUM_PASSENGERS = "NumPassengers";
std::string const OcrNameplates::CLASSNAME_VEHICLE_MODEL = "VehicleModel";
std::string const OcrNameplates::CLASSNAME_ENGINE_DISPLACEMENT = "EngineDisplacement";
std::string const OcrNameplates::CLASSNAME_DATE_OF_MANUFACTURE = "DateOfManufacture";
std::string const OcrNameplates::CLASSNAME_PAINT = "Paint";

std::unordered_map<std::string ,std::string> const OcrNameplates::CLASSNAME_ENG_TO_CHN = {
        {CLASSNAME_VIN, "车辆识别代号"},
        {CLASSNAME_MANUFACTURER, "制造商"},
        {CLASSNAME_BRAND, "品牌"},
        {CLASSNAME_MAX_MASS_ALLOWED, "最大允许总质量"},
        {CLASSNAME_MAX_NET_POWER_OF_ENGINE, "发动机最大净功率"},
        {CLASSNAME_COUNTRY, "制造国"},
        {CLASSNAME_FACTORY, "生产厂名"},
        {CLASSNAME_ENGINE_MODEL, "发动机型号"},
        {CLASSNAME_NUM_PASSENGERS, "乘坐人数"},
        {CLASSNAME_VEHICLE_MODEL, "整车型号"},
        {CLASSNAME_ENGINE_DISPLACEMENT, "发动机排量"},
        {CLASSNAME_DATE_OF_MANUFACTURE, "制造年月"},
        {CLASSNAME_PAINT, "涂料"}
};

OcrNameplates::~OcrNameplates() = default;

OcrNameplates::OcrNameplates() = default;

InfoTable const& OcrNameplates::getResult() const {
    return _result;
}

void OcrNameplates::printResultToConsoleInChinese() const {
    _result.printResultToConsole([](std::string const& key) {
        auto itrKeyMapped = CLASSNAME_ENG_TO_CHN.find(key);
        return itrKeyMapped == CLASSNAME_ENG_TO_CHN.end() ?
               "" : itrKeyMapped->second;
    });
}

void OcrNameplates::printResultToFileInChinese(std::ofstream& outFile) const {
    _result.printResultToFile(outFile, [](std::string const& key) {
        auto itrKeyMapped = CLASSNAME_ENG_TO_CHN.find(key);
        return itrKeyMapped == CLASSNAME_ENG_TO_CHN.end() ?
               "" : itrKeyMapped->second;
    });
}