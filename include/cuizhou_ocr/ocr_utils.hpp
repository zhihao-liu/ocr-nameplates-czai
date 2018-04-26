//
// Created by Zhihao Liu on 18-4-4.
//

#ifndef CUIZHOU_OCR_OCRUTILS_H
#define CUIZHOU_OCR_OCRUTILS_H

#include <vector>
#include <string>


namespace cuizhou {

std::vector<std::string> readClassNames(std::string const& path, bool addBackground = false);
bool isNumbericChar(std::string const& str);

} // end namespace cuizhou


#endif //CUIZHOU_OCR_OCRUTILS_H
