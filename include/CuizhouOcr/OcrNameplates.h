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

        InfoTable _result;

        OcrNameplates();
    };
}


#endif //OCR_CUIZHOU_OCRNAMEPLATES_H
