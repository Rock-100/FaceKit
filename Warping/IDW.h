#ifndef __IDW__
#define __IDW__

#include <iostream>
#include <cstdio>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>

#include "opencv2/opencv.hpp"

#define EPS 1e-7

class IDW
{
public:
    IDW();
    void SetStartControlPoint(std::vector<cv::Point> s);
    void SetEndControlPoint(std::vector<cv::Point> e);
    cv::Mat Transform(cv::Mat input);

private:
    double Distance(cv::Point &p, cv::Point &q);
    std::vector<double> GetControlPointWeight(cv::Point input);
    cv::Point GetTransformPoint(cv::Point input);

    double weight_;
    std::vector<cv::Point> startControlPoint_;
    std::vector<cv::Point> endControlPoint_;

};

#endif
