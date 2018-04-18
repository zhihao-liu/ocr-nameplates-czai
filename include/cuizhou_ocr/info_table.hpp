//
// Created by Zhihao Liu on 18-4-4.
//

#ifndef OCR_CUIZHOU_INFOTABLE_H
#define OCR_CUIZHOU_INFOTABLE_H

#include <map>
#include <fstream>
#include <iostream>
#include <opencv2/core/core.hpp>


namespace cuizhou {
    class DetectedItem {
    public:
        std::string content;
        cv::Rect rect;

        ~DetectedItem() = default;
        DetectedItem() = default;
        DetectedItem(std::string const& _content, cv::Rect const& _rect);
    };

    class KeyValuePair {
    public:
        DetectedItem key;
        DetectedItem value;

        ~KeyValuePair() = default;
        KeyValuePair() = default;
        KeyValuePair(DetectedItem const& _key, DetectedItem const& _value);
    };

    class InfoTable {
    public:
        ~InfoTable() = default;
        InfoTable() = default;
        void clear();
        void put(std::string const& keyName, KeyValuePair const& keyValuePair);
        KeyValuePair const* get(std::string const& keyName) const;

        template<typename F> void printResultToConsole(F const& keyMapped) const;
        template<typename F> void printResultToFile(std::ofstream& outFile, F const& keyMapped) const;

    private:
        std::map<std::string, KeyValuePair> _table;
    };

    template<typename F>
    void InfoTable::printResultToConsole(F const& keyMapped) const {
        std::cout << std::endl;
        for (auto const& pair: _table) {
            std::string keyMappedStr = keyMapped(pair.first);
            std::cout << keyMappedStr << ": " << pair.second.value.content << std::endl;
        }
        std::cout << std::endl;
    }

    template<typename F>
    void InfoTable::printResultToFile(std::ofstream& outFile, F const& keyMapped) const {
        for (auto const& pair: _table) {
            std::string keyMappedStr = keyMapped(pair.first);
            outFile << keyMappedStr << ": " << pair.second.value.content << std::endl;
        }
    }
}

#endif //OCR_CUIZHOU_INFOTABLE_H
