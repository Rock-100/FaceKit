#!/usr/bin/python3
from ctypes import *
import cv2
import numpy as np
import sys
import os
from ipdb import set_trace as dbg
from enum import IntEnum

class CPoint(Structure):
    _fields_ = [("x", c_int),
                ("y", c_int)]

FEAT_POINTS = 14
class CWindow(Structure):
    _fields_ = [("x", c_int),
                ("y", c_int),
                ("width", c_int),
                ("angle", c_int),
                ("score", c_float),
                ("points",CPoint*14)]

class FeatEnam(IntEnum):
    CHIN_0 = 0
    CHIN_1 = 1
    CHIN_2 = 2
    CHIN_3 = 3
    CHIN_4 = 4
    CHIN_5 = 5
    CHIN_6 = 6
    CHIN_7 = 7
    CHIN_8 = 8
    NOSE = 9
    EYE_LEFT = 10
    EYE_RIGHT = 11
    MOUTH_LEFT = 12
    MOUTH_RIGHT = 13
    FEAT_POINTS = 14


lib = CDLL("libPCN.so", RTLD_GLOBAL)

init_detector = lib.init_detector
#void *init_detector(const char *detection_model_path, 
#            const char *pcn1_proto, const char *pcn2_proto, const char *pcn3_proto, 
#            const char *tracking_model_path, const char *tracking_proto,
#            int min_face_size, float pyramid_scale_factor, float detection_thresh_stage1,
#            float detection_thresh_stage2, float detection_thresh_stage3, int tracking_period,
#            float tracking_thresh, int do_smooth)
init_detector.argtypes = [
        c_char_p, c_char_p, c_char_p, 
        c_char_p, c_char_p, c_char_p,
        c_int,c_float,c_float,c_float,
        c_float,c_int,c_float,c_int]
init_detector.restype = c_void_p

#CWindow* detect_faces(void* pcn, unsigned char* raw_img,size_t rows, size_t cols, int *lwin)
detect_faces = lib.detect_faces
detect_faces.argtypes = [c_void_p, POINTER(c_ubyte),c_size_t,c_size_t,POINTER(c_int)]
detect_faces.restype = POINTER(CWindow)

#void free_faces(CWindow* wins)
free_faces = lib.free_faces
free_faces.argtypes= [c_void_p]

# void free_detector(void *pcn)
free_detector = lib.free_detector
free_detector.argtypes= [c_void_p]

def DrawFace(win,img):
    x1 = win.x;
    y1 = win.y;
    x2 = win.width + win.x - 1;
    y2 = win.width + win.y - 1;
    centerX = (x1 + x2) / 2;
    centerY = (y1 + y2) / 2;
    angle = win.angle
    R = cv2.getRotationMatrix2D((centerX,centerY),angle,1)
    pts = np.array([[x1,y1,1],[x1,y2,1],[x2,y2,1],[x2,y1,1]], np.int32)
    pts = (pts @ R.T).astype(int) #Rotate points
    pts = pts.reshape((-1,1,2))
    cv2.polylines(img,[pts],True,(0,0,255))
    cv2.putText(img,"Face",(x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2,cv2.LINE_AA)

def DrawPoints(win,img):
    f = FeatEnam.NOSE
    cv2.circle(img,(win.points[f].x,win.points[f].y),2,(255, 153, 255))
    f = FeatEnam.EYE_LEFT
    cv2.circle(img,(win.points[f].x,win.points[f].y),2,(0,255,255))
    f = FeatEnam.EYE_RIGHT
    cv2.circle(img,(win.points[f].x,win.points[f].y),2,(0,255,255))
    f = FeatEnam.MOUTH_LEFT
    cv2.circle(img,(win.points[f].x,win.points[f].y),2,(0,255,0))
    f = FeatEnam.MOUTH_RIGHT
    cv2.circle(img,(win.points[f].x,win.points[f].y),2,(0,255,0))

def SetThreadCount(threads):
    os.environ['OMP_NUM_THREADS'] = str(threads)

def c_str(str_in):
    return c_char_p(str_in.encode('utf-8'))

if __name__=="__main__":

    SetThreadCount(1)
    if len(sys.argv)==2:
        cap = cv2.VideoCapture(sys.argv[1])
    else:
        cap = cv2.VideoCapture(0)
    #cap = cv2.VideoCapture('udp://127.0.0.1:2234',cv2.CAP_FFMPEG)
    detection_model_path = c_str("./model/PCN.caffemodel")
    pcn1_proto = c_str("./model/PCN-1.prototxt")
    pcn2_proto = c_str("./model/PCN-2.prototxt")
    pcn3_proto = c_str("./model/PCN-3.prototxt")
    tracking_model_path = c_str("./model/PCN-Tracking.caffemodel")
    tracking_proto = c_str("./model/PCN-Tracking.prototxt")

    detector = init_detector(detection_model_path,pcn1_proto,pcn2_proto,pcn3_proto,
			tracking_model_path,tracking_proto, 
			15,1.45,0.5,0.5,0.98,30,0.9,1)

    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) # float
    fps = cap.get(cv2.CAP_PROP_FPS) # float
    while cap.isOpened():
        ret, frame = cap.read()
        if frame.shape[0] == 0:
            break
        face_count = c_int(0)
        raw_data = frame.ctypes.data_as(POINTER(c_ubyte))
        windows = detect_faces(detector, raw_data, 
                int(height), int(width),
                pointer(face_count))
        for i in range(face_count.value):
            DrawFace(windows[i],frame)
            DrawPoints(windows[i],frame)
        free_faces(windows)

        cv2.imshow('window', frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    free_detector(detector)

