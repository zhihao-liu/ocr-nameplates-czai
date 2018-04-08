//
// Created by Zhihao Liu on 18-4-4.
//

#ifndef OCR_CUIZHOU_OCRNAMEPLATES_H
#define OCR_CUIZHOU_OCRNAMEPLATES_H

#include "OcrHandler.h"
#include "InfoTable.h"


namespace cuizhou {
    class OcrNameplates : public OcrHandler {
    public:
        virtual ~OcrNameplates() override;
        virtual void processImage() override = 0;
        InfoTable const& getResult() const;

    protected:
        static std::string const CLASSNAME_VIN;
        static std::string const CLASSNAME_MANUFACTURER;
        static std::string const CLASSNAME_BRAND;
        static std::string const CLASSNAME_MAX_MASS_ALLOWED;
        static std::string const CLASSNAME_MAX_NET_POWER_OF_ENGINE;
        static std::string const CLASSNAME_COUNTRY;
        static std::string const CLASSNAME_FACTORY;
        static std::string const CLASSNAME_ENGINE_MODEL;
        static std::string const CLASSNAME_NUM_PASSENGERS;
        static std::string const CLASSNAME_VEHICLE_MODEL;
        static std::string const CLASSNAME_ENGINE_DISPLACEMENT;
        static std::string const CLASSNAME_DATE_OF_MANUFACTURE;
        static std::string const CLASSNAME_PAINT;

        InfoTable _result;

        OcrNameplates();
    };
}


#endif //OCR_CUIZHOU_OCRNAMEPLATES_H
