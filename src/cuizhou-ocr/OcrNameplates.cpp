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

OcrNameplates::~OcrNameplates() = default;

OcrNameplates::OcrNameplates() = default;

InfoTable const& OcrNameplates::getResult() const {
    return _result;
}