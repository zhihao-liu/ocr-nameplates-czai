//
// Created by Zhihao Liu on 3/28/18.
//

#include <iostream>
#include <fstream>
#include <dirent.h>
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

using namespace std;
using namespace cv;

int main(int argc, char *argv[]) {
    string srcDir = "/Users/liuzhihao/Dropbox/Advancement/Internship Cuizhou/OCR/Labeling AlfaRomeo/full-size/img/";
    string dstDir = "/Users/liuzhihao/Desktop/cropped/img/";
    string infoDir = "/Users/liuzhihao/Desktop/cropped/info/";

    DIR* dir;
    struct dirent* de;

    dir = opendir(srcDir.c_str());
    while ((de = readdir(dir)) != nullptr) {
        string fileName = de->d_name;
        if (fileName.length() < 4 || fileName.substr(fileName.length() - 4, fileName.length()) != ".jpg") continue;
        string imgName = fileName.substr(0, fileName.length() - 4);

        string imgPath = srcDir + fileName;
        Mat imgRgb = imread(imgPath);
        Mat img = imgRgb.clone();
        cvtColor(img, img, CV_RGB2GRAY);

        GaussianBlur(img, img, Size(3, 3), 0.51);
        Canny(img, img, 250, 100);

        int gapRadius = 6;
        int const numberOfLine = 8;
        int indexOfLine[numberOfLine];

        for (int k = 0; k < numberOfLine; k++) {
            float mostSum = 0;

            for (int i = gapRadius; i < img.rows - gapRadius; i += gapRadius) {
                float sumValueOfGap = 0;
                for (int j = -gapRadius; j <= gapRadius; j++)
                    sumValueOfGap += sum(img.row(i + j))[0];

                if (sumValueOfGap > mostSum) {
                    mostSum = sumValueOfGap;
                    indexOfLine[k] = i;
                }

            }

            for (int j = -gapRadius; j <= gapRadius; j++)
                img.row(indexOfLine[k] + j).setTo(0);

        }

        sort(indexOfLine, indexOfLine + numberOfLine);
        int centerY = indexOfLine[numberOfLine / 2 - 1] + indexOfLine[numberOfLine / 2];
        centerY /= 2;

        int windowRadius = img.rows / 6;
        int top = centerY - windowRadius;
        int bot = centerY + windowRadius;
        if (top < 0) top = 0;
        if (bot >img.rows - 1) bot = img.rows - 1;

        vector<Rect> rois = {
                Rect(Point(0, top), Size(img.cols * 2 / 3, bot - top + 1)),
                Rect(Point(img.cols / 3, top), Size(img.cols * 2 / 3, bot - top + 1))
        };

        for (int i = 0; i < rois.size(); ++i) {
            Rect const& roi = rois[i];
            string outputName = imgName + "-cropped-" + to_string(i);

            Mat imgCropped = imgRgb(roi);
            imwrite(dstDir + outputName + ".jpg", imgCropped);

            ofstream fout(infoDir + outputName + ".txt");
            fout << roi.tl().x << endl;
            fout << roi.tl().y << endl;
            fout << roi.br().x << endl;
            fout << roi.br().y << endl;
            fout.close();
        }
    }
    closedir(dir);

    cout << "DONE" << endl;
    return 0;
}