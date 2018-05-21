***New***

 - 2018.05.21: A fast version of PCN with smaller networks is released.


# Real-Time Rotation-Invariant Face Detection with Progressive Calibration Networks

Progressive Calibration Networks (PCN) is an accurate rotation-invariant face detector running at real-time speed on CPU, published in CVPR 2018. This is a binary library for PCN. In this implementation, we don't use network quantization or compression, and the program runs on CPU with a single thread.

### Results

Some detection results can be viewed in the following illustrations:

<img src='result/demo.png' width=800 height=410>

<img src='result/1.gif' width=800 height=460>

<img src='result/2.gif' width=800 height=460>

PCN is designed aiming for low time-cost. We compare PCN's speed with other rotation-invariant face detectors' on standard VGA images(640x480) with 40x40 minimum face size. The detectors run on a desktop computer with 3.4GHz CPU, GTX Titan X. The speed results together with the recall rate at 100 false positives on multi-oriented FDDB are shown in the following table. Detailed experiment settings can be found in our paper.

<img src='result/result.png' width=800 height=150>

### Usage

Set minimum size of faces to detect (`size` >= 20)

- `detector.SetMinFaceSize(size);`
  
Set scaling factor of image pyramid (1.4 <= `factor` <= 1.6)
  
- `detector.SetImagePyramidScaleFactor(factor);`
  
Set score threshold of detected faces (0 <= `thresh` <= 1)
  
- `detector.SetScoreThresh(thresh);`

Smooth the face boxes or not (smooth = true or false, recommend using it on video to get stabler face boxes)
  
- `detector.SetVideoSmooth(smooth);`

See [picture.cpp](code/picture.cpp) and [video.cpp](code/video.cpp) for details. If you want to reproduce the results on FDDB, set `size` and `factor` as 20 and 1.414 respectively, or you can adjust them according to your application.

Compile and run the demo:
```Shell
cd $PCN_ROOT/code/PCN or cd $PCN_ROOT/code/FastPCN
# You should set "CAFFEROOT" in compile.sh and run.sh first. 
sh compile.sh picture/video
sh run.sh picture/video
```

### Links

* [arxiv](https://arxiv.org/pdf/1804.06039.pdf)
* [ppt](PCN.pdf)

### Prerequisites

* Linux
* Caffe (recommend using OpenBLAS, compile caffe mannully or refer to [this docker image](https://github.com/Edwardmark/caffe-docker))
* OpenCV (2.4.10, or other compatible version)


### License

This code is distributed under the [BSD 2-Clause license](LICENSE).

### Citing PCN

If you find PCN useful in your research, please consider citing:

    @inproceedings{shiCVPR18pcn,
        Author = {Xuepeng Shi and Shiguang Shan and Meina Kan and Shuzhe Wu and Xilin Chen},
        Title = {Real-Time Rotation-Invariant Face Detection with Progressive Calibration Networks},
        Booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
        Year = {2018}
    }

### Contact

Xuepeng Shi, xuepeng.shi@vipl.ict.ac.cn
