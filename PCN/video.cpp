#include "PCN.h"

int main()
{
    PCN* detector = (PCN*) init_detector(30,1.414,0.37,0.43,0.97,30,0.95,1,0.9,0.6);
    cv::VideoCapture capture(0);
    cv::Mat img;
    cv::TickMeter tm;
    while (1)
    {
        capture >> img;
        tm.reset();
        tm.start();
        //std::vector<Window> faces = detector.Detect(img);
        //std::vector<Window> faces = detector->DetectTrack(img);
	
	Window *wins= NULL;
	size_t lwins = 0;
	wins = detect_faces(detector,img.data,img.rows, img.cols, &lwins);

        tm.stop();
        int fps = 1000.0 / tm.getTimeMilli();
        std::stringstream ss;
        ss << std::setw(4) << fps;
        cv::putText(img, std::string("PCN:") + ss.str() + "FPS",
                    cv::Point(20, 45), 4, 1, cv::Scalar(0, 0, 125));
        for (int i = 0; i < lwins; i++)
        {
            DrawFace(img, wins[i]);
        }
	free(wins);
        cv::imshow("PCN", img);
        if (cv::waitKey(1) == 'q')
            break;
    }

    capture.release();
    cv::destroyAllWindows();
    free_detector(detector);
    return 0;
}
