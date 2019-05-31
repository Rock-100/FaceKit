#!/usr/bin/python3
from ctypes import *
import cv2
import numpy as np
import sys

class Window(Structure):
    _fields_ = [("x", c_int),
                ("y", c_int),
                ("width", c_int),
                ("angle", c_int),
                ("score", c_float),
                ("points14",c_void_p)]

lib = CDLL("libPCN.so", RTLD_GLOBAL)

init_detector = lib.init_detector
#	void *init_detector(int min_face_size, float pyramid_scale_factor, float detection_thresh_stage1,
#			float detection_thresh_stage2, float detection_thresh_stage3, int tracking_period,
#			float tracking_thresh, int do_smooth, float iou_high_thresh, float iou_low_thresh)
init_detector.argtypes = [c_int,c_float,c_float,c_float,c_float,c_int,c_float,c_int,c_float, c_float]
init_detector.restype = c_void_p

#Window* detect_faces(void* pcn, unsigned char* raw_img,size_t rows, size_t cols, size_t *lwin)
detect_faces = lib.detect_faces
detect_faces.argtypes = [c_void_p, POINTER(c_ubyte),c_size_t,c_size_t,POINTER(c_size_t)]
detect_faces.restype = POINTER(Window)

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
    pts = np.array([[x1,y1],[x1,y2],[x2,y2],[x2,y1]], np.int32)
    pts = pts.reshape((-1,1,2))
    cv2.polylines(img,[pts],True,(0,0,255))
    cv2.putText(img,"emil",(x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2,cv2.LINE_AA)

if __name__=="__main__":
    #cap = cv2.VideoCapture('udp://127.0.0.1:2234',cv2.CAP_FFMPEG)
    if len(sys.argv)==2:
        cap = cv2.VideoCapture(sys.argv[1])
    else:
        cap = cv2.VideoCapture(0)

    detector = init_detector(25,1.414,0.37,0.43,0.97,30,0.95,1,0.9,0.6)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) # float
    fps = cap.get(cv2.CAP_PROP_FPS) # float
    while cap.isOpened():
        ret, frame = cap.read()
        if frame.shape[0] == 0:
            break

        raw_data = frame.ctypes.data_as(POINTER(c_ubyte))
        n_win = c_size_t(0)
        windows = detect_faces(detector, raw_data, int(height), int(width), pointer(n_win))

        for i in range(n_win.value):
            DrawFace(windows[i],frame)


        cv2.imshow('window', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    free_detector(detector)

