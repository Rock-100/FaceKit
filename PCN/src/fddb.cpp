#include "PCN.h"

int main()
{
    std::string dataPath = "/home/jack/Desktop/fddb/originalPics/";
    std::ofstream out("detList.txt");
    std::ifstream in("/home/jack/Desktop/fddb/image_fold-total_noext.txt");
    std::string line;
    int total = 0;
    float time = 0;

    PCN detector("model/PCN.caffemodel",
                 "model/PCN-1.prototxt", "model/PCN-2.prototxt", "model/PCN-3.prototxt",
                 "model/PCN-Tracking.caffemodel",
                 "model/PCN-Tracking.prototxt");
    /// detection
    detector.SetMinFaceSize(20);
    detector.SetImagePyramidScaleFactor(1.414);
    detector.SetDetectionThresh(0.37, 0.43, 0.7);
    /// tracking
    detector.SetTrackingPeriod(30);
    detector.SetTrackingThresh(0.95);
    detector.SetVideoSmooth(false);

    while (std::getline(in, line))
    {
        std::cout << line << std::endl;
        cv::Mat img = cv::imread(dataPath + line + ".jpg");

        long t0 = cv::getTickCount();
        std::vector<Window> faces = detector.Detect(img);
        long t1 = cv::getTickCount();
        time += (t1 - t0) / cv::getTickFrequency();
        total += 1;
        std::cout << "count: " << total << "  ave time: " <<
                  time / total << "s" << std::endl;

        out << line << std::endl;
        out << faces.size() << std::endl;
        for (int i = 0; i < faces.size(); i++)
        {
            if (abs(faces[i].angle) < 45 or abs(faces[i].angle) > 135)
                out << faces[i].x << " " << faces[i].y - 0.1 * faces[i].width << " " <<
                    faces[i].width << " " << faces[i].width * 1.2 << " " <<
                    faces[i].score << std::endl;
            else
                out << faces[i].x - 0.1 * faces[i].width << " " << faces[i].y << " " <<
                    faces[i].width * 1.2 << " " << faces[i].width << " " <<
                    faces[i].score << std::endl;
        }
    }
    in.close();
    out.close();
    return 0;
}
