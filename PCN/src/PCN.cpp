#include "PCN.h"
#ifndef MODEL_PATH
#define MODEL_PATH "./model"
#endif
struct Window2
{
    int x, y, w, h;
    float angle, scale, conf;
    int id;
    Window2(int x_, int y_, int w_, int h_, float a_, float s_, float c_, int id_)
        : x(x_), y(y_), w(w_), h(h_), angle(a_), scale(s_), conf(c_), id(id_)
    {}
};

class Impl
{
public:
    void LoadModel(std::string modelDetect, std::string net1, std::string net2, std::string net3,
                   std::string modelTrack, std::string netTrack);
    cv::Mat ResizeImg(cv::Mat img, float scale);
    static bool CompareWin(const Window2 &w1, const Window2 &w2);
    bool Legal(int x, int y, cv::Mat img);
    bool Inside(int x, int y, Window2 rect);
    int SmoothAngle(int a, int b);
    std::vector<Window2> SmoothWindow(std::vector<Window2> aList,std::vector<Window2> bList);
    float IoU(Window2 &w1, Window2 &w2);
    std::vector<Window2> NMS(std::vector<Window2> &winList, bool local, float threshold);
    std::vector<Window2> DeleteFP(std::vector<Window2> &winList);
    cv::Mat PreProcessImg(cv::Mat img);
    cv::Mat PreProcessImg(cv::Mat img,  int dim);
    void SetInput(cv::Mat input, caffe::shared_ptr<caffe::Net<float> > &net);
    void SetInput(std::vector<cv::Mat> &input, caffe::shared_ptr<caffe::Net<float> > &net);
    cv::Mat PadImg(cv::Mat img);
    std::vector<Window> TransWindow(cv::Mat img, cv::Mat imgPad, std::vector<Window2> &winList);
    std::vector<Window2> Stage1(cv::Mat img, cv::Mat imgPad, caffe::shared_ptr<caffe::Net<float> > &net, float thres);
    std::vector<Window2> Stage2(cv::Mat img, cv::Mat img180,
                                caffe::shared_ptr<caffe::Net<float> > &net, float thres, int dim, std::vector<Window2> &winList);
    std::vector<Window2> Stage3(cv::Mat img, cv::Mat img180, cv::Mat img90, cv::Mat imgNeg90,
                                caffe::shared_ptr<caffe::Net<float> > &net, float thres, int dim, std::vector<Window2> &winList);
    std::vector<Window2> Detect(cv::Mat img, cv::Mat imgPad);
    std::vector<Window2> Track(cv::Mat img, caffe::shared_ptr<caffe::Net<float> > &net,
                               float thres, int dim, std::vector<Window2> &winList);

    std::vector<Window2> preList;
    caffe::shared_ptr<caffe::Net<float> > net_[4];
    int minFace_;
    float scale_;
    int stride_;
    float classThreshold_[3];
    float nmsThreshold_[3];
    float angleRange_;
    bool stable_;
    int period_;
    float trackThreshold_;
    float augScale_;
    float smooth_high_threshold_;
    float smooth_low_threshold_;
    int global_id_;
};

PCN::PCN(std::string modelDetect, std::string net1, std::string net2, std::string net3,
         std::string modelTrack, std::string netTrack) : impl_(new Impl())
{
    Impl *p = (Impl *)impl_;
    p->LoadModel(modelDetect, net1, net2, net3, modelTrack, netTrack);
    p->global_id_ = 0;
}

void PCN::SetVideoSmooth(bool stable)
{
    Impl *p = (Impl *)impl_;
    p->stable_ = stable;
}

void PCN::SetMinFaceSize(int minFace)
{
    Impl *p = (Impl *)impl_;
    p->minFace_ = minFace > 20 ? minFace : 20;
    p->minFace_ *= 1.4;
}

void PCN::SetIOUThresh(float high_thresh, float low_thresh)
{
    Impl *p = (Impl *)impl_;
    p->smooth_high_threshold_ = high_thresh;
    p->smooth_low_threshold_ = low_thresh;
}

void PCN::SetDetectionThresh(float thresh1, float thresh2, float thresh3)
{
    Impl *p = (Impl *)impl_;
    p->classThreshold_[0] = thresh1;
    p->classThreshold_[1] = thresh2;
    p->classThreshold_[2] = thresh3;
    p->nmsThreshold_[0] = 0.8;
    p->nmsThreshold_[1] = 0.8;
    p->nmsThreshold_[2] = 0.3;
    p->stride_ = 8;
    p->angleRange_ = 45;
    p->augScale_ = 0.15;
}

void PCN::SetImagePyramidScaleFactor(float factor)
{
    Impl *p = (Impl *)impl_;
    p->scale_ = factor;
}

void PCN::SetTrackingPeriod(int period)
{
    Impl *p = (Impl *)impl_;
    p->period_ = period;
}

void PCN::SetTrackingThresh(float thres)
{
    Impl *p = (Impl *)impl_;
    p->trackThreshold_ = thres;
}

std::vector<Window> PCN::Detect(cv::Mat img)
{
    Impl *p = (Impl *)impl_;
    cv::Mat imgPad = p->PadImg(img);
    std::vector<Window2> winList = p->Detect(img, imgPad);

    if (p->stable_)
    {
        winList = p->SmoothWindow(winList, p->preList);
    }
    return p->TransWindow(img, imgPad, winList);
}

std::vector<Window> PCN::DetectTrack(cv::Mat img)
{
    Impl *p = (Impl *)impl_;
    cv::Mat imgPad = p->PadImg(img);

    static int detectFlag = p->period_;
    std::vector<Window2> winList = p->Track(imgPad, p->net_[3], p->trackThreshold_, 72, p->preList);
    winList = p->NMS(winList, false, p->nmsThreshold_[2]);
    winList = p->DeleteFP(winList);
    if (p->stable_)
        winList = p->SmoothWindow(winList,p->preList);

    if (detectFlag == p->period_)
    {
        std::vector<Window2> tmpList = p->Detect(img, imgPad);
	tmpList = p->NMS(tmpList, false, p->nmsThreshold_[2]);
	tmpList = p->DeleteFP(tmpList);
        winList = p->SmoothWindow(tmpList,winList);//recooncile tracking with detection
    }

    p->preList = winList;
    detectFlag--;
    if (detectFlag == 0)
        detectFlag = p->period_;
    return p->TransWindow(img, imgPad, winList);
}


// Implementation functions
void Impl::LoadModel(std::string modelDetect, std::string net1, std::string net2, std::string net3,
                     std::string modelTrack, std::string netTrack)
{
    caffe::Caffe::set_mode(caffe::Caffe::CPU);
    google::InitGoogleLogging("VR");
    FLAGS_logtostderr = 0;

    net_[0].reset(new caffe::Net<float>(net1.c_str(), caffe::TEST));
    net_[0]->CopyTrainedLayersFrom(modelDetect.c_str());
    net_[1].reset(new caffe::Net<float>(net2.c_str(), caffe::TEST));
    net_[1]->CopyTrainedLayersFrom(modelDetect.c_str());
    net_[2].reset(new caffe::Net<float>(net3.c_str(), caffe::TEST));
    net_[2]->CopyTrainedLayersFrom(modelDetect.c_str());

    net_[3].reset(new caffe::Net<float>(netTrack.c_str(), caffe::TEST));
    net_[3]->CopyTrainedLayersFrom(modelTrack.c_str());

    google::ShutdownGoogleLogging();
}

cv::Mat Impl::PreProcessImg(cv::Mat img)
{
    cv::Mat mean(img.size(), CV_32FC3, cv::Scalar(104, 117, 123));
    cv::Mat imgF;
    img.convertTo(imgF, CV_32FC3);
    return imgF - mean;
}

cv::Mat Impl::PreProcessImg(cv::Mat img, int dim)
{
    cv::Mat imgNew;
    cv::resize(img, imgNew, cv::Size(dim, dim));
    cv::Mat mean(imgNew.size(), CV_32FC3, cv::Scalar(104, 117, 123));
    cv::Mat imgF;
    imgNew.convertTo(imgF, CV_32FC3);
    return imgF - mean;
}

void Impl::SetInput(cv::Mat input, caffe::shared_ptr<caffe::Net<float> > &net)
{
    int rows = input.rows, cols = input.cols;
    int length = rows * cols;
    caffe::Blob<float>* inputBlobs = net->input_blobs()[0];
    inputBlobs->Reshape(1, 3, rows, cols);
    net->Reshape();
    std::vector<cv::Mat> tmp;
    cv::split(input, tmp);
    float *p = inputBlobs->mutable_cpu_data();
    for (int i = 0; i < tmp.size(); i++)
    {
        memcpy(p, tmp[i].data, sizeof(float) * length);
        p += length;
    }
}

void Impl::SetInput(std::vector<cv::Mat> &input, caffe::shared_ptr<caffe::Net<float> > &net)
{
    int rows = input[0].rows, cols = input[0].cols;
    int length = rows * cols;
    caffe::Blob<float>* inputBlobs = net->input_blobs()[0];
    inputBlobs->Reshape(input.size(), 3, rows, cols);
    net->Reshape();
    float *p = inputBlobs->mutable_cpu_data();
    std::vector<cv::Mat> tmp;
    for (int i = 0; i < input.size(); i++)
    {
        cv::split(input[i], tmp);
        for (int j = 0; j < tmp.size(); j++)
        {
            memcpy(p, tmp[j].data, sizeof(float) * length);
            p += length;
        }
    }
}

cv::Mat Impl::ResizeImg(cv::Mat img, float scale)
{
    cv::Mat ret;
    cv::resize(img, ret, cv::Size(int(img.cols / scale), int(img.rows / scale)));
    return ret;
}

bool Impl::CompareWin(const Window2 &w1, const Window2 &w2)
{
    return w1.conf > w2.conf;
}

bool Impl::Legal(int x, int y, cv::Mat img)
{
    if (x >= 0 && x < img.cols && y >= 0 && y < img.rows)
        return true;
    else
        return false;
}

bool Impl::Inside(int x, int y, Window2 rect)
{
    if (x >= rect.x && y >= rect.y && x < rect.x + rect.w && y < rect.y + rect.h)
        return true;
    else
        return false;
}

int Impl::SmoothAngle(int a, int b)
{
    if (a > b)
        std::swap(a, b);
    int diff = (b - a) % 360;
    if (diff < 180)
        return a + diff / 2;
    else
        return b + (360 - diff) / 2;
}

float Impl::IoU(Window2 &w1, Window2 &w2)
{
    int xOverlap = std::max(0, std::min(w1.x + w1.w - 1, w2.x + w2.w - 1) - std::max(w1.x, w2.x) + 1);
    int yOverlap = std::max(0, std::min(w1.y + w1.h - 1, w2.y + w2.h - 1) - std::max(w1.y, w2.y) + 1);
    int intersection = xOverlap * yOverlap;
    int unio = w1.w * w1.h + w2.w * w2.h - intersection;
    return float(intersection) / unio;
}

std::vector<Window2> Impl::NMS(std::vector<Window2> &winList, bool local, float threshold)
{
    if (winList.size() == 0)
        return winList;
    std::sort(winList.begin(), winList.end(), CompareWin);
    bool flag[winList.size()];
    memset(flag, 0, winList.size());
    for (int i = 0; i < winList.size(); i++)
    {
        if (flag[i])
            continue;
        for (int j = i + 1; j < winList.size(); j++)
        {
            if (local && abs(winList[i].scale - winList[j].scale) > EPS)
                continue;
            if (IoU(winList[i], winList[j]) > threshold)
                flag[j] = 1;
        }
    }
    std::vector<Window2> ret;
    for (int i = 0; i < winList.size(); i++)
    {
        if (!flag[i]) ret.push_back(winList[i]);
    }
    return ret;
}

/// to delete some false positives
std::vector<Window2> Impl::DeleteFP(std::vector<Window2> &winList)
{
    if (winList.size() == 0)
        return winList;
    std::sort(winList.begin(), winList.end(), CompareWin);
    bool flag[winList.size()];
    memset(flag, 0, winList.size());
    for (int i = 0; i < winList.size(); i++)
    {
        if (flag[i])
            continue;
        for (int j = i + 1; j < winList.size(); j++)
        {
            if (Inside(winList[j].x, winList[j].y, winList[i]) && Inside(winList[j].x + winList[j].w - 1, winList[j].y + winList[j].h - 1, winList[i]))
                flag[j] = 1;
        }
    }
    std::vector<Window2> ret;
    for (int i = 0; i < winList.size(); i++)
    {
        if (!flag[i]) ret.push_back(winList[i]);
    }
    return ret;
}

/// to detect faces on the boundary
cv::Mat Impl::PadImg(cv::Mat img)
{
    int row = std::min(int(img.rows * 0.2), 100);
    int col = std::min(int(img.cols * 0.2), 100);
    cv::Mat ret;
    cv::copyMakeBorder(img, ret, row, row, col, col, cv::BORDER_CONSTANT);
    return ret;
}

std::vector<Window2> Impl::Stage1(cv::Mat img, cv::Mat imgPad, caffe::shared_ptr<caffe::Net<float> > &net, float thres)
{
    int row = (imgPad.rows - img.rows) / 2;
    int col = (imgPad.cols - img.cols) / 2;
    std::vector<Window2> winList;
    int netSize = 24;
    float curScale;
    curScale = minFace_ / float(netSize);
    cv::Mat imgResized = ResizeImg(img, curScale);
    while (std::min(imgResized.rows, imgResized.cols) >= netSize)
    {
        SetInput(PreProcessImg(imgResized), net);
        net->Forward();
        caffe::Blob<float>* reg = net->output_blobs()[0];
        caffe::Blob<float>* prob = net->output_blobs()[1];
        caffe::Blob<float>* rotateProb = net->output_blobs()[2];
        float w = netSize * curScale;
        for (int i = 0; i < prob->height(); i++)
        {
            for (int j = 0; j < prob->width(); j++)
            {
                if (prob->data_at(0, 1, i, j) > thres)
                {
                    float sn = reg->data_at(0, 0, i, j);
                    float xn = reg->data_at(0, 1, i, j);
                    float yn = reg->data_at(0, 2, i, j);
                    int rx = int(j * curScale * stride_ - 0.5 * sn * w + sn * xn * w + 0.5 * w) + col;
                    int ry = int(i * curScale * stride_ - 0.5 * sn * w + sn * yn * w + 0.5 * w) + row;
                    int rw = int(w * sn);
                    if (Legal(rx, ry, imgPad) && Legal(rx + rw - 1, ry + rw - 1, imgPad))
                    {
                        if (rotateProb->data_at(0, 1, i, j) > 0.5)
                            winList.push_back(Window2(rx, ry, rw, rw, 0, curScale, prob->data_at(0, 1, i, j),j));
                        else
                            winList.push_back(Window2(rx, ry, rw, rw, 180, curScale, prob->data_at(0, 1, i, j),j));
                    }
                }
            }
        }
        imgResized = ResizeImg(imgResized, scale_);
        curScale = float(img.rows) / imgResized.rows;
    }
    return winList;
}

std::vector<Window2> Impl::Stage2(cv::Mat img, cv::Mat img180, caffe::shared_ptr<caffe::Net<float> > &net, float thres, int dim, std::vector<Window2> &winList)
{
    if (winList.size() == 0)
        return winList;
    std::vector<cv::Mat> dataList;
    int height = img.rows;
    for (int i = 0; i < winList.size(); i++)
    {
        if (abs(winList[i].angle) < EPS)
            dataList.push_back(PreProcessImg(img(cv::Rect(winList[i].x, winList[i].y, winList[i].w, winList[i].h)), dim));
        else
        {
            int y2 = winList[i].y + winList[i].h - 1;
            dataList.push_back(PreProcessImg(img180(cv::Rect(winList[i].x, height - 1 - y2, winList[i].w, winList[i].h)), dim));
        }
    }
    SetInput(dataList, net);
    net->Forward();
    caffe::Blob<float>* reg = net->output_blobs()[0];
    caffe::Blob<float>* prob = net->output_blobs()[1];
    caffe::Blob<float>* rotateProb = net->output_blobs()[2];
    std::vector<Window2> ret;
    for (int i = 0; i < winList.size(); i++)
    {
        if (prob->data_at(i, 1, 0, 0) > thres)
        {
            float sn = reg->data_at(i, 0, 0, 0);
            float xn = reg->data_at(i, 1, 0, 0);
            float yn = reg->data_at(i, 2, 0, 0);
            int cropX = winList[i].x;
            int cropY = winList[i].y;
            int cropW = winList[i].w;
            if (abs(winList[i].angle)  > EPS)
                cropY = height - 1 - (cropY + cropW - 1);
            int w = int(sn * cropW);
            int x = int(cropX  - 0.5 * sn * cropW + cropW * sn * xn + 0.5 * cropW);
            int y = int(cropY  - 0.5 * sn * cropW + cropW * sn * yn + 0.5 * cropW);
            float maxRotateScore = 0;
            int maxRotateIndex = 0;
            for (int j = 0; j < 3; j++)
            {
                if (rotateProb->data_at(i, j, 0, 0) > maxRotateScore)
                {
                    maxRotateScore = rotateProb->data_at(i, j, 0, 0);
                    maxRotateIndex = j;
                }
            }
            if (Legal(x, y, img) && Legal(x + w - 1, y + w - 1, img))
            {
                float angle = 0;
                if (abs(winList[i].angle)  < EPS)
                {
                    if (maxRotateIndex == 0)
                        angle = 90;
                    else if (maxRotateIndex == 1)
                        angle = 0;
                    else
                        angle = -90;
                    ret.push_back(Window2(x, y, w, w, angle, winList[i].scale, prob->data_at(i, 1, 0, 0),i));
                }
                else
                {
                    if (maxRotateIndex == 0)
                        angle = 90;
                    else if (maxRotateIndex == 1)
                        angle = 180;
                    else
                        angle = -90;
                    ret.push_back(Window2(x, height - 1 -  (y + w - 1), w, w, angle, winList[i].scale, prob->data_at(i, 1, 0, 0),i));
                }
            }
        }
    }
    return ret;
}

std::vector<Window2> Impl::Stage3(cv::Mat img, cv::Mat img180, cv::Mat img90, cv::Mat imgNeg90, caffe::shared_ptr<caffe::Net<float> > &net, float thres, int dim, std::vector<Window2> &winList)
{
    if (winList.size() == 0)
        return winList;
    std::vector<cv::Mat> dataList;
    int height = img.rows;
    int width = img.cols;
    for (int i = 0; i < winList.size(); i++)
    {
        if (abs(winList[i].angle) < EPS)
            dataList.push_back(PreProcessImg(img(cv::Rect(winList[i].x, winList[i].y, winList[i].w, winList[i].h)), dim));
        else if (abs(winList[i].angle - 90) < EPS)
        {
            dataList.push_back(PreProcessImg(img90(cv::Rect(winList[i].y, winList[i].x, winList[i].h, winList[i].w)), dim));
        }
        else if (abs(winList[i].angle + 90) < EPS)
        {
            int x = winList[i].y;
            int y = width - 1 - (winList[i].x + winList[i].w - 1);
            dataList.push_back(PreProcessImg(imgNeg90(cv::Rect(x, y, winList[i].w, winList[i].h)), dim));
        }
        else
        {
            int y2 = winList[i].y + winList[i].h - 1;
            dataList.push_back(PreProcessImg(img180(cv::Rect(winList[i].x, height - 1 - y2, winList[i].w, winList[i].h)), dim));
        }
    }
    SetInput(dataList, net);
    net->Forward();
    caffe::Blob<float>* reg = net->output_blobs()[0];
    caffe::Blob<float>* prob = net->output_blobs()[1];
    caffe::Blob<float>* rotateProb = net->output_blobs()[2];
    std::vector<Window2> ret;
    for (int i = 0; i < winList.size(); i++)
    {
        if (prob->data_at(i, 1, 0, 0) > thres)
        {
            float sn = reg->data_at(i, 0, 0, 0);
            float xn = reg->data_at(i, 1, 0, 0);
            float yn = reg->data_at(i, 2, 0, 0);
            int cropX = winList[i].x;
            int cropY = winList[i].y;
            int cropW = winList[i].w;
            cv::Mat imgTmp = img;
            if (abs(winList[i].angle - 180)  < EPS)
            {
                cropY = height - 1 - (cropY + cropW - 1);
                imgTmp = img180;
            }
            else if (abs(winList[i].angle - 90)  < EPS)
            {
                std::swap(cropX, cropY);
                imgTmp = img90;
            }
            else if (abs(winList[i].angle + 90)  < EPS)
            {
                cropX = winList[i].y;
                cropY = width - 1 - (winList[i].x + winList[i].w - 1);
                imgTmp = imgNeg90;
            }

            int w = int(sn * cropW);
            int x = int(cropX  - 0.5 * sn * cropW + cropW * sn * xn + 0.5 * cropW);
            int y = int(cropY  - 0.5 * sn * cropW + cropW * sn * yn + 0.5 * cropW);
            float angle = angleRange_ * rotateProb->data_at(i, 0, 0, 0);
            if (Legal(x, y, imgTmp) && Legal(x + w - 1, y + w - 1, imgTmp))
            {
                if (abs(winList[i].angle)  < EPS)
                    ret.push_back(Window2(x, y, w, w, angle, winList[i].scale, prob->data_at(i, 1, 0, 0),i));
                else if (abs(winList[i].angle - 180)  < EPS)
                {
                    ret.push_back(Window2(x, height - 1 -  (y + w - 1), w, w, 180 - angle, winList[i].scale, prob->data_at(i, 1, 0, 0),i));

                }
                else if (abs(winList[i].angle - 90)  < EPS)
                {
                    ret.push_back(Window2(y, x, w, w, 90 - angle, winList[i].scale, prob->data_at(i, 1, 0, 0),i));
                }
                else
                {
                    ret.push_back(Window2(width - y - w, x, w, w, -90 + angle, winList[i].scale, prob->data_at(i, 1, 0, 0),i));
                }
            }
        }
    }
    return ret;
}

std::vector<Window> Impl::TransWindow(cv::Mat img, cv::Mat imgPad, std::vector<Window2> &winList)
{
    int row = (imgPad.rows - img.rows) / 2;
    int col = (imgPad.cols - img.cols) / 2;

    std::vector<Window> ret;
    for(int i = 0; i < winList.size(); i++)
    {
        if (winList[i].w > 0 && winList[i].h > 0)
            ret.push_back(Window(winList[i].x - col, winList[i].y - row, winList[i].w, winList[i].angle, winList[i].conf, winList[i].id));
    }
    return ret;
}

std::vector<Window2> Impl::SmoothWindow(std::vector<Window2> aList, std::vector<Window2> bList)
{
    //static std::vector<Window2> bList;
    std::vector<Window2> newWinList;
    for (int i = 0; i < aList.size(); i++)
    {
	    int new_id = 1;
        for (int j = 0; j < bList.size(); j++)
        {
            if (IoU(aList[i], bList[j]) > smooth_high_threshold_)
            {
                aList[i].conf = (aList[i].conf + bList[j].conf) / 2;
                aList[i].x = bList[j].x;
                aList[i].y = bList[j].y;
                aList[i].w = bList[j].w;
                aList[i].h = bList[j].h;
                aList[i].angle = bList[j].angle;
		aList[i].id = bList[j].id;
		new_id = 0;
            }
            else if (IoU(aList[i], bList[j]) > smooth_low_threshold_)
            {
                aList[i].conf = (aList[i].conf + bList[j].conf) / 2;
                aList[i].x = (aList[i].x + bList[j].x) / 2;
                aList[i].y = (aList[i].y + bList[j].y) / 2;
                aList[i].w = (aList[i].w + bList[j].w) / 2;
                aList[i].h = (aList[i].h + bList[j].h) / 2;
                aList[i].angle = SmoothAngle(aList[i].angle, bList[j].angle);
		aList[i].id = bList[j].id;
		new_id = 0;
            }
        }
	if (new_id > 0){
		aList[i].id = global_id_++; //consistent id after smoothing
	}
    }
    //bList = aList;
    return aList;
}

std::vector<Window2> Impl::Detect(cv::Mat img, cv::Mat imgPad)
{
    cv::Mat img180, img90, imgNeg90;
    cv::flip(imgPad, img180, 0);
    cv::transpose(imgPad, img90);
    cv::flip(img90, imgNeg90, 0);

    std::vector<Window2> winList = Stage1(img, imgPad, net_[0], classThreshold_[0]);
    winList = NMS(winList, true, nmsThreshold_[0]);

    winList = Stage2(imgPad, img180, net_[1], classThreshold_[1], 24, winList);
    winList = NMS(winList, true, nmsThreshold_[1]);

    winList = Stage3(imgPad, img180, img90, imgNeg90, net_[2], classThreshold_[2], 48, winList);
    winList = NMS(winList, false, nmsThreshold_[2]);
    winList = DeleteFP(winList);
    return winList;
}

std::vector<Window2> Impl::Track(cv::Mat img, caffe::shared_ptr<caffe::Net<float> > &net,
                                 float thres, int dim, std::vector<Window2> &winList)
{
    if (winList.size() == 0)
        return winList;
    std::vector<cv::Mat> dataList;
    for (int i = 0; i < winList.size(); i++)
    {
        Window win(winList[i].x - augScale_ * winList[i].w, winList[i].y - augScale_ * winList[i].w,
                   winList[i].w + 2 * augScale_ * winList[i].w, winList[i].angle, winList[i].conf, winList[i].id);
        dataList.push_back(PreProcessImg(CropFace(img, win, dim), dim));
    }
    SetInput(dataList, net);
    net->Forward();
    caffe::Blob<float>* reg = net->output_blobs()[0];
    caffe::Blob<float>* prob = net->output_blobs()[1];
    caffe::Blob<float>* rotateProb = net->output_blobs()[2];
    std::vector<Window2> ret;
    for (int i = 0; i < winList.size(); i++)
    {
        if (prob->data_at(i, 1, 0, 0) > thres)
        {
            float sn = reg->data_at(i, 0, 0, 0);
            float xn = reg->data_at(i, 1, 0, 0);
            float yn = reg->data_at(i, 2, 0, 0);
            int cropX = winList[i].x - augScale_ * winList[i].w;
            int cropY = winList[i].y - augScale_ * winList[i].w;
            int cropW = winList[i].w + 2 * augScale_ * winList[i].w;
            float theta = -winList[i].angle * M_PI / 180;
            int w = int(sn * cropW);
            int x = int(cropX  - 0.5 * sn * cropW +
                        cropW * sn * xn * std::cos(theta) - cropW * sn * yn * std::sin(theta) + 0.5 * cropW);
            int y = int(cropY  - 0.5 * sn * cropW +
                        cropW * sn * xn * std::sin(theta) + cropW * sn * yn * std::cos(theta) + 0.5 * cropW);
            float angle = angleRange_ * rotateProb->data_at(i, 0, 0, 0);
            if (Legal(x, y, img) && Legal(x + w - 1, y + w - 1, img))
            {
                int tmpW = w / (1 + 2 * augScale_);
                if (tmpW >= 20)
                    ret.push_back(Window2(x + augScale_ * tmpW, y + augScale_ * tmpW,
                                          tmpW, tmpW, winList[i].angle + angle, winList[i].scale, prob->data_at(i, 1, 0, 0),i));
            }
        }
    }
    return ret;
}



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
    cv::putText(img, std::string("id:") + std::to_string(face.id),
		    cv::Point(x1, y1), 2, 1, cv::Scalar(255, 0, 0));
}

cv::Mat CropFace(cv::Mat img, Window face, int cropSize)
{
    int x1 = face.x;
    int y1 = face.y;
    int x2 = face.width + face.x - 1;
    int y2 = face.width + face.y - 1;
    int centerX = (x1 + x2) / 2;
    int centerY = (y1 + y2) / 2;
    cv::Point2f srcTriangle[3];
    cv::Point2f dstTriangle[3];
    srcTriangle[0] = RotatePoint(x1, y1, centerX, centerY, face.angle);
    srcTriangle[1] = RotatePoint(x1, y2, centerX, centerY, face.angle);
    srcTriangle[2] = RotatePoint(x2, y2, centerX, centerY, face.angle);
    dstTriangle[0] = cv::Point(0, 0);
    dstTriangle[1] = cv::Point(0, cropSize - 1);
    dstTriangle[2] = cv::Point(cropSize - 1, cropSize - 1);
    cv::Mat rotMat = cv::getAffineTransform(srcTriangle, dstTriangle);
    cv::Mat ret;
    cv::warpAffine(img, ret, rotMat, cv::Size(cropSize, cropSize));
    return ret;
}

//C API stuff
void *init_detector(int min_face_size, float pyramid_scale_factor, float detection_thresh_stage1,
		float detection_thresh_stage2, float detection_thresh_stage3, int tracking_period,
		float tracking_thresh, int do_smooth, float iou_high_thresh, float iou_low_thresh)
{
    PCN *detector = new PCN(MODEL_PATH"/PCN.caffemodel",
                 MODEL_PATH"/PCN-1.prototxt", MODEL_PATH"/PCN-2.prototxt", MODEL_PATH"/PCN-3.prototxt",
                 MODEL_PATH"/PCN-Tracking.caffemodel",
                 MODEL_PATH"/PCN-Tracking.prototxt");
    /// detection
    detector->SetMinFaceSize(min_face_size);
    detector->SetImagePyramidScaleFactor(pyramid_scale_factor);
    detector->SetDetectionThresh(detection_thresh_stage1,detection_thresh_stage2,detection_thresh_stage3);
    /// tracking
    detector->SetTrackingPeriod(tracking_period);
    detector->SetTrackingThresh(tracking_thresh);
    detector->SetVideoSmooth((bool)do_smooth);
    detector->SetIOUThresh(iou_high_thresh,iou_low_thresh);
    return static_cast<void*> (detector);
}

Window* detect_faces(void* pcn, unsigned char* raw_img,size_t rows, size_t cols, size_t *lwin)
{
	PCN* detector = (PCN*) pcn;
	cv::Mat img(rows,cols, CV_8UC3, (void*)raw_img);
        std::vector<Window> faces = detector->DetectTrack(img);
	*lwin = faces.size();
	Window* wins = (Window*)malloc( sizeof(Window)*(*lwin));
	std::copy( faces.begin(), faces.end(), wins);
	return wins;
}

void free_detector(void *pcn)
{
	PCN* detector = (PCN*) pcn;
	delete detector;
}

