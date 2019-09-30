import cv2
import numpy as np
import time



def chokan(img):
    img1=img
    height=240
    width=320
    image = np.zeros((700, 700, 3), np.uint8)
    src = np.array([[100,40],[210,40],[0,239],[width,239]],np.float32)
    dst = np.array([[0,0],[500,0],[int(500/5),800],[int(500/5*4),800]],np.float32)

    M = cv2.getPerspectiveTransform(src, dst)
    warp = cv2.warpPerspective(img1.copy(), M, (500, 800))

    return warp
    #cv2.imshow('transform', warp)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    #cv2.imwrite("convert.png",warp)


fractal_controll=0
minLineLength = 1000
maxLineGap = 1
lines=0
defcount=0
Xpoint=145
avecontrast=0
contraststate=0
fps=30
Height=240
Width=320


cap = cv2.VideoCapture("../output2.m4v")
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v') 
convert_out = cv2.VideoWriter('output2.m4v',fourcc, fps, (500,800))



while(cap.isOpened()):

    #img1 = cv2.imread("frame.png", cv2.IMREAD_COLOR)
    # cv2.imshow('transform', img1)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    ret,img1= cap.read()
    if ret==False:
        break
    out=chokan(img1)
    convert_out.write(out)
convert_out.release()