//
// Created by Zhihao Liu on 18-4-4.
//

#ifndef OCR_CUIZHOU_INFOTABLE_H
#define OCR_CUIZHOU_INFOTABLE_H

#include <map>
#include "KeyValuePair.h"


namespace cuizhou {
    class InfoTable {
    public:
        ~InfoTable();
        InfoTable();
        void put(std::string const& keyName, KeyValuePair const& keyValuePair);
        KeyValuePair const& get(std::string const& keyName);

    private:
        std::map<std::string, KeyValuePair> _table;
    };
}


#endif //OCR_CUIZHOU_INFOTABLE_H
