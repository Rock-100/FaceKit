#include "PCN.h"

int main()
{
    PCN detector("model/PCN.caffemodel",
                 "model/PCN-1.prototxt", "model/PCN-2.prototxt", "model/PCN-3.prototxt",
                 "model/PCN-Tracking.caffemodel",
                 "model/PCN-Tracking.prototxt");
    /// detection
    detector.SetMinFaceSize(45);
    detector.SetImagePyramidScaleFactor(1.414);
    detector.SetDetectionThresh(0.37, 0.43, 0.97);
    /// tracking
    detector.SetTrackingPeriod(20);
    detector.SetTrackingThresh(0.95);
    detector.SetVideoSmooth(true);

    cv::VideoCapture capture(0);
    cv::Mat img;
    cv::TickMeter tm;
    while (1)
    {
        capture >> img;
        tm.reset();
        tm.start();
        //std::vector<Window> faces = detector.Detect(img);
        std::vector<Window> faces = detector.DetectTrack(img);
        tm.stop();
        int fps = 1000.0 / tm.getTimeMilli();
        std::stringstream ss;
        ss << std::setw(4) << fps;
        cv::putText(img, std::string("PCN:") + ss.str() + "FPS",
                    cv::Point(20, 45), 4, 1, cv::Scalar(0, 0, 125));
        for (int i = 0; i < faces.size(); i++)
        {
            DrawFace(img, faces[i]);
        }
        cv::imshow("PCN", img);
        if (cv::waitKey(1) == 'q')
            break;
    }

    capture.release();
    cv::destroyAllWindows();

    return 0;
}
