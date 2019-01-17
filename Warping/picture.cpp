#include "IDW.h"

int main()
{
    cv::Mat input = cv::imread("imgs/1.jpg");

    std::vector<cv::Point> startPoint
    {
        cv::Point(133, 313), cv::Point(139, 345), cv::Point(163, 378), cv::Point(195, 400), cv::Point(244, 412),
        cv::Point(305, 398), cv::Point(340, 376), cv::Point(361, 336), cv::Point(366, 307)
    };
    std::vector<cv::Point> endPoint
    {
        cv::Point(140, 313), cv::Point(150, 345), cv::Point(175, 378), cv::Point(205, 400), cv::Point(244, 412),
        cv::Point(290, 398), cv::Point(328, 376), cv::Point(350, 336), cv::Point(360, 307)
    };

    for (int i = 0; i <= 5; i++)
    {
        startPoint.push_back(cv::Point(0, input.rows * i / 5));
        endPoint.push_back(cv::Point(0, input.rows * i / 5));

        startPoint.push_back(cv::Point(input.cols - 1, input.rows * i / 5));
        endPoint.push_back(cv::Point(input.cols - 1, input.rows * i / 5));
    }


    IDW idw;
    idw.SetStartControlPoint(startPoint);
    idw.SetEndControlPoint(endPoint);

    cv::TickMeter tm;
    tm.start();
    cv::Mat output = idw.Transform(input);
    tm.stop();
    std::cout << "FaceWarping: " << tm.getTimeMilli() << "ms" << std::endl;

    for(int i = 0; i < startPoint.size(); i++)
    {
        cv::line(input, startPoint[i], endPoint[i], cv::Scalar(255, 255, 0), 3);
        cv::circle(input, endPoint[i], 5, cv::Scalar(255, 0, 0), -1);

        cv::line(output, startPoint[i], endPoint[i], cv::Scalar(255, 255, 0), 3);
        cv::circle(output, endPoint[i], 5, cv::Scalar(255, 0, 0), -1);

    }
    cv::imshow("Input", input);
    cv::imshow("Output", output);

    cv::waitKey();
    cv::destroyAllWindows();

    return 0;
}
