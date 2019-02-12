#include "PCN.h"

int main()
{
    PCN detector("model/PCN.caffemodel",
                 "model/PCN-1.prototxt", "model/PCN-2.prototxt", "model/PCN-3.prototxt",
                 "model/PCN-Tracking.caffemodel",
                 "model/PCN-Tracking.prototxt");
    /// detection
    detector.SetMinFaceSize(20);
    detector.SetImagePyramidScaleFactor(1.414);
    detector.SetDetectionThresh(0.37, 0.43, 0.97);
    /// tracking
    detector.SetTrackingPeriod(30);
    detector.SetTrackingThresh(0.95);
    detector.SetVideoSmooth(false);

    for (int i = 0; i <= 26; i++)
    {
        cv::Mat img =
            cv::imread("imgs/" + std::to_string(i) + ".jpg");
        cv::TickMeter tm;
        tm.reset();
        tm.start();
        std::vector<Window> faces = detector.Detect(img);
        tm.stop();
        std::cout << "Image: " << i << std::endl;
        std::cout << "Time Cost: "<<
                  tm.getTimeMilli() << " ms" << std::endl;
        for (int j = 0; j < faces.size(); j++)
        {
            DrawFace(img, faces[j]);
        }
        cv::imshow("PCN", img);
        cv::waitKey();
    }

    cv::destroyAllWindows();

    return 0;
}
