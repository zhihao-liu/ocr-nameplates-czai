//
// Created by Zhihao Liu on 18-4-4.
//

#include "ocr_implementation/ocr_nameplate.h"
#include <opencv2/imgproc/imgproc.hpp>


namespace cuizhou {

ClassnameDict<OcrNameplate::NameplateField> const OcrNameplate::fieldDict_(
        {NameplateField::VIN, NameplateField::MANUFACTURER, NameplateField::BRAND, NameplateField::MAX_MASS_ALLOWED, NameplateField::MAX_NET_POWER_OF_ENGINE, NameplateField::COUNTRY, NameplateField::FACTORY, NameplateField::ENGINE_MODEL, NameplateField::NUM_PASSENGERS, NameplateField::VEHICLE_MODEL, NameplateField::ENGINE_DISPLACEMENT, NameplateField::DATE_OF_MANUFACTURE, NameplateField::PAINT},
        NameplateField::UNKNOWN,
        {"Vin", "Manufacturer", "Brand", "MaxMassAllowed", "MaxNetPowerOfEngine", "Country", "Factory", "EngineModel", "NumPassengers", "VehicleModel", "EngineDisplacement", "DateOfManufacture", "Paint"},
        std::string(),
        {"车辆识别代号", "制造商", "品牌", "最大允许总质量", "发动机最大净功率", "制造国", "生产厂名", "发动机型号", "乘坐人数", "整车型号", "发动机排量", "制造年月", "涂料"},
        std::string()
);

OcrNameplate::~OcrNameplate() = default;

OcrNameplate::OcrNameplate() = default;

std::vector<KeyValueDetection> OcrNameplate::getResultAsArray() const {
    std::vector<KeyValueDetection> resultVector;
    std::transform(result_.cbegin(), result_.cend(), std::back_inserter(resultVector),
                   [](std::pair<NameplateField const, KeyValueDetection> const& elem) { return elem.second; });
    return resultVector;
}

cv::Mat OcrNameplate::drawResult() const {
    cv::Mat imgToShow = image_.clone();
    cv::Scalar clrKeyRect(0, 0, 255);
    cv::Scalar clrValueRect(0, 255, 255);

    for (auto const& item : result_) {
        cv::rectangle(imgToShow, item.second.key.rect, clrKeyRect);
        cv::rectangle(imgToShow, item.second.value.rect, clrValueRect);
    }

    return imgToShow;
}

std::string OcrNameplate::getResultAsString() const {
    std::stringstream ss;
    for (auto const& item : getResultAsArray()) {
        ss << item << std::endl;
    }
    return ss.str();
}

} // end namespace cuizhou
