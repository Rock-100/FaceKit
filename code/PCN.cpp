#include "PCN.h"

struct Window2
{
    int x, y, w, h;
    float angle, scale, conf;
    Window2(int x_, int y_, int w_, int h_, float a_, float s_, float c_)
        : x(x_), y(y_), w(w_), h(h_), angle(a_), scale(s_), conf(c_)
    {}
};

class Impl
{
public:
    void LoadModel(std::string model, std::string net1, std::string net2, std::string net3);
    cv::Mat ResizeImg(cv::Mat img, float scale);
    static bool CompareWin(const Window2 &w1, const Window2 &w2);
    bool Legal(int x, int y, cv::Mat img);
    bool Inside(int x, int y, Window2 rect);
    int SmoothAngle(int a, int b);
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


    caffe::shared_ptr<caffe::Net<float> > net_[3];
    int minFace_;
    float scale_;
    int stride_;
    float classThreshold_[3];
    float nmsThreshold_[3];
    float angleRange_;
    bool stable_;
};

PCN::PCN(std::string model, std::string net1, std::string net2, std::string net3) : impl_(new Impl())
{
    Impl *p = (Impl *)impl_;
    p->LoadModel(model, net1, net2, net3);
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

void PCN::SetScoreThresh(float thresh1, float thresh2, float thresh3)
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
}

void PCN::SetImagePyramidScaleFactor(float factor)
{
    Impl *p = (Impl *)impl_;
    p->scale_ = factor;
}

std::vector<Window> PCN::DetectFace(cv::Mat img)
{
    Impl *p = (Impl *)impl_;
    cv::Mat imgPad = p->PadImg(img);
    cv::Mat img180, img90, imgNeg90;
    cv::flip(imgPad, img180, 0);
    cv::transpose(imgPad, img90);
    cv::flip(img90, imgNeg90, 0);

    std::vector<Window2> winList = p->Stage1(img, imgPad, p->net_[0], p->classThreshold_[0]);
    winList = p->NMS(winList, true, p->nmsThreshold_[0]);

    winList = p->Stage2(imgPad, img180, p->net_[1], p->classThreshold_[1], 24, winList);
    winList = p->NMS(winList, true, p->nmsThreshold_[1]);

    winList = p->Stage3(imgPad, img180, img90, imgNeg90, p->net_[2], p->classThreshold_[2], 48, winList);
    winList = p->NMS(winList, false, p->nmsThreshold_[2]);
    winList = p->DeleteFP(winList);

    static std::vector<Window2> preList;
    if (p->stable_)
    {
        for (int i = 0; i < winList.size(); i++)
        {
            for (int j = 0; j < preList.size(); j++)
            {
                if (p->IoU(winList[i], preList[j]) > 0.9)
                    winList[i] = preList[j];
                else if (p->IoU(winList[i], preList[j]) > 0.6)
                {
                    winList[i].x = (winList[i].x + preList[j].x) / 2;
                    winList[i].y = (winList[i].y + preList[j].y) / 2;
                    winList[i].w = (winList[i].w + preList[j].w) / 2;
                    winList[i].h = (winList[i].h + preList[j].h) / 2;
                    winList[i].angle = p->SmoothAngle(winList[i].angle, preList[j].angle);
                }
            }
        }
        preList = winList;
    }
    return p->TransWindow(img, imgPad, winList);
}

void Impl::LoadModel(std::string model,
                     std::string net1, std::string net2, std::string net3)
{
    caffe::Caffe::set_mode(caffe::Caffe::CPU);
    google::InitGoogleLogging("VR");
    FLAGS_logtostderr = 0;

    net_[0].reset(new caffe::Net<float>(net1.c_str(), caffe::TEST));
    net_[0]->CopyTrainedLayersFrom(model.c_str());
    net_[1].reset(new caffe::Net<float>(net2.c_str(), caffe::TEST));
    net_[1]->CopyTrainedLayersFrom(model.c_str());
    net_[2].reset(new caffe::Net<float>(net3.c_str(), caffe::TEST));
    net_[2]->CopyTrainedLayersFrom(model.c_str());
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
                            winList.push_back(Window2(rx, ry, rw, rw, 0, curScale, prob->data_at(0, 1, i, j)));
                        else
                            winList.push_back(Window2(rx, ry, rw, rw, 180, curScale, prob->data_at(0, 1, i, j)));
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
                    ret.push_back(Window2(x, y, w, w, angle, winList[i].scale, prob->data_at(i, 1, 0, 0)));
                }
                else
                {
                    if (maxRotateIndex == 0)
                        angle = 90;
                    else if (maxRotateIndex == 1)
                        angle = 180;
                    else
                        angle = -90;
                    ret.push_back(Window2(x, height - 1 -  (y + w - 1), w, w, angle, winList[i].scale, prob->data_at(i, 1, 0, 0)));
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
                    ret.push_back(Window2(x, y, w, w, angle, winList[i].scale, prob->data_at(i, 1, 0, 0)));
                else if (abs(winList[i].angle - 180)  < EPS)
                {
                    ret.push_back(Window2(x, height - 1 -  (y + w - 1), w, w, 180 - angle, winList[i].scale, prob->data_at(i, 1, 0, 0)));
                }
                else if (abs(winList[i].angle - 90)  < EPS)
                {
                    ret.push_back(Window2(y, x, w, w, 90 - angle, winList[i].scale, prob->data_at(i, 1, 0, 0)));
                }
                else
                {
                    ret.push_back(Window2(width - y - w, x, w, w, -90 + angle, winList[i].scale, prob->data_at(i, 1, 0, 0)));
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
            ret.push_back(Window(winList[i].x - col, winList[i].y - row, winList[i].w, winList[i].angle, winList[i].conf));
    }
    return ret;
}


