#ifndef __PCN__
#define __PCN__

#include <iostream>
#include <cstdio>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>

#include "opencv2/opencv.hpp"
#include "caffe/caffe.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

struct Window
{
    int x, y, width;
    float angle, score;
    Window(int x_, int y_, int w_, float a_, float s_) : x(x_), y(y_), width(w_), angle(a_), score(s_)
    {}
};

cv::Point RotatePoint(int x, int y, float centerX, float centerY, float angle)
{
    x -= centerX;
    y -= centerY;
    float theta = -angle * M_PI / 180;
    int rx = int(centerX + x * std::cos(theta) - y * std::sin(theta));
    int ry = int(centerY + x * std::sin(theta) + y * std::cos(theta));
    return cv::Point(rx, ry);
}

void DrawLine(cv::Mat img, std::vector<cv::Point> pointList)
{
    int thick = 2;
    CvScalar cyan = CV_RGB(0, 255, 255);
    CvScalar blue = CV_RGB(0, 0, 255);
    cv::line(img, pointList[0], pointList[1], cyan, thick);
    cv::line(img, pointList[1], pointList[2], cyan, thick);
    cv::line(img, pointList[2], pointList[3], cyan, thick);
    cv::line(img, pointList[3], pointList[0], blue, thick);
}

void DrawFace(cv::Mat img, Window face)
{
    int x1 = face.x;
    int y1 = face.y;
    int x2 = face.width + face.x - 1;
    int y2 = face.width + face.y - 1;
    int centerX = (x1 + x2) / 2;
    int centerY = (y1 + y2) / 2;
    std::vector<cv::Point> pointList;
    pointList.push_back(RotatePoint(x1, y1, centerX, centerY, face.angle));
    pointList.push_back(RotatePoint(x1, y2, centerX, centerY, face.angle));
    pointList.push_back(RotatePoint(x2, y2, centerX, centerY, face.angle));
    pointList.push_back(RotatePoint(x2, y1, centerX, centerY, face.angle));
    DrawLine(img, pointList);
}

class PCN
{
public:
    PCN(std::string model);
    void SetMinFaceSize(int minFace);
    void SetScoreThresh(float thresh);
    void SetImagePyramidScaleFactor(float factor);
    void SetVideoSmooth(bool smooth);
    std::vector<Window> DetectFace(cv::Mat img);

private:
    void* impl_;
};

#endif
