//
// Created by Zhihao Liu on 18-4-4.
//

#ifndef OCR_CUIZHOU_INFOTABLE_H
#define OCR_CUIZHOU_INFOTABLE_H

#include <map>
#include "opencv2/core/core.hpp"


namespace cuizhou {
    struct DetectedItem {
        std::string const content;
        cv::Rect const rect;
    };

    struct KeyValuePair {
        DetectedItem const key;
        DetectedItem const value;
    };

    class InfoTable {
    public:
        ~InfoTable();
        InfoTable();
        void put(std::string const& keyName, KeyValuePair const& keyValuePair);
        KeyValuePair const* get(std::string const& keyName) const;

    private:
        std::map<std::string, KeyValuePair> _table;
    };
}


#endif //OCR_CUIZHOU_INFOTABLE_H
