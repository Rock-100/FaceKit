#include "IDW.h"

IDW::IDW()
{
    weight_ = 2;
}

double IDW::Distance(cv::Point &p, cv::Point &q)
{
    cv::Point diff = p - q;
    return cv::sqrt(diff.x * diff.x + diff.y * diff.y);
}

void IDW::SetStartControlPoint(std::vector<cv::Point> s)
{
    startControlPoint_ = s;
}

void IDW::SetEndControlPoint(std::vector<cv::Point> e)
{
    endControlPoint_ = e;
}

cv::Point IDW::GetTransformPoint(cv::Point input)
{
    double x = 0, y = 0;
    std::vector<double> weightMap = GetControlPointWeight(input);
    for (int i = 0; i < startControlPoint_.size(); i++)
    {
        double offsetX = startControlPoint_[i].x - endControlPoint_[i].x;
        double offsetY = startControlPoint_[i].y - endControlPoint_[i].y;
        x += offsetX * weightMap[i];
        y += offsetY * weightMap[i];
    }
    return cv::Point(input.x + x, input.y + y);
}

std::vector<double> IDW::GetControlPointWeight(cv::Point input)
{
    std::vector<double> weightMap;
    double weightSum = 0;
    for (int i = 0; i < startControlPoint_.size(); i++)
    {
        double temp = 1 / (Distance(endControlPoint_[i], input) + EPS);
        temp = pow(temp, weight_);
        weightSum = weightSum + temp;
        weightMap.push_back(temp);
    }
    for (int i = 0; i < startControlPoint_.size(); i++)
    {
        weightMap[i] /= weightSum;
    }
    return weightMap;
}

cv::Mat IDW::Transform(cv::Mat input)
{
    cv::Mat output(input.rows, input.cols, input.type());
    for (int j = 0; j < output.rows; j++)
    {
        for (int i = 0; i < output.cols; i++)
        {
            cv::Point temp = GetTransformPoint(cv::Point(i, j));
            if (temp.x >= 0 && temp.y >= 0 && temp.x < output.cols && temp.y < output.rows)
            {
                output.at<cv::Vec3b>(j, i)[0] = input.at<cv::Vec3b>(temp.y, temp.x)[0];
                output.at<cv::Vec3b>(j, i)[1] = input.at<cv::Vec3b>(temp.y, temp.x)[1];
                output.at<cv::Vec3b>(j, i)[2] = input.at<cv::Vec3b>(temp.y, temp.x)[2];
            }
        }
    }
    return output;
}
