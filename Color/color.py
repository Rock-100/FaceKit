import cv2
import numpy as np
import copy

def color_trans(ref_img, in_img):
    ref_img = copy.deepcopy(ref_img)
    in_img = copy.deepcopy(in_img)
    ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2LAB)
    in_img = cv2.cvtColor(in_img, cv2.COLOR_BGR2LAB)
    ref_avg, ref_std = cv2.meanStdDev(ref_img)
    in_avg, in_std = cv2.meanStdDev(in_img)
    height, width, channel = in_img.shape
    for i in range(0, height):
        for j in range(0, width):
            for k in range(0, channel):
                v = in_img[i, j, k]
                v = (v - in_avg[k]) * (ref_std[k] / in_std[k]) + ref_avg[k]
                v = 0 if v < 0 else v
                v = 255 if v > 255 else v
                in_img[i, j, k] = v
    out_img = cv2.cvtColor(in_img, cv2.COLOR_LAB2BGR)
    return out_img


img1 = cv2.imread("imgs/1.png")
img2 = cv2.imread("imgs/2.png")
cv2.imshow("1", img1)
cv2.imshow("2", img2)
cv2.imshow("1to2", color_trans(img1, img2))
cv2.imshow("2to1", color_trans(img2, img1))
cv2.waitKey()
cv2.destroyAllWindows()
