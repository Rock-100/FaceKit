#include "Common.h"
#include "BilateralFilter.h"
#include "GuideFilter.h"

int main()
{
    cv::Mat msrcImage = cv::imread("imgs/2.jpg");
    cv::TickMeter tm;
    Image srcImage;
    srcImage.data = (unsigned char *)malloc(msrcImage.rows * msrcImage.cols * msrcImage.channels() * sizeof(unsigned char));
    Mat2Image(msrcImage, srcImage);
    ShowImage("Input", srcImage);

    Image resImage;
    resImage.data = (unsigned char *)malloc(msrcImage.rows * msrcImage.cols * msrcImage.channels() * sizeof(unsigned char));
    tm.reset();
    tm.start();
    BilateralFilter(srcImage, resImage, 5, 60, 170);
    tm.stop();
    std::cout << "BilateralFilter: " << tm.getTimeMilli() << "ms" << std::endl;
    ShowImage("BilateralFilter", resImage);

    Image resImage2;
    resImage2.data = (unsigned char *)malloc(msrcImage.rows * msrcImage.cols * msrcImage.channels() * sizeof(unsigned char));
    tm.reset();
    tm.start();
    GuideFilter(srcImage, resImage2, 5, 0.005);
    tm.stop();
    std::cout << "GuideFilter: " << tm.getTimeMilli() << "ms" << std::endl;
    ShowImage("GuideFilter", resImage2);

    cv::waitKey();
    cv::destroyAllWindows();
    free(srcImage.data);
    free(resImage.data);
    free(resImage2.data);

    return 0;
}
