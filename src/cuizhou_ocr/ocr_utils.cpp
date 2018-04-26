//
// Created by Zhihao Liu on 18-4-4.
//

#include "ocr_utils.hpp"
#include <fstream>


namespace cuizhou {

std::vector<std::string> readClassNames(std::string const& path, bool addBackground) {
    std::ifstream file(path);
    std::vector<std::string> classNames;
    if (addBackground) classNames.emplace_back("__background__");

    std::string line;
    while (getline(file, line)) {
        classNames.push_back(std::move(line));
    }
    return classNames;
};

bool isNumbericChar(std::string const& str) {
    return str.length() == 1 && str[0] >= '0' && str[0] <= '9';
}

} // end namespace cuizhou