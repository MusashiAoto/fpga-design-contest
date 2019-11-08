import cv2
import numpy as np
import time




fractal_controll=0
minLineLength = 1000
maxLineGap = 1
lines=0
defcount=0
Xpoint=145
avecontrast=0
contraststate=0
fps=30
Height=800
Width=500


def chokan(img):
    img1=img
    width=320

    #image = np.zeros((700, 700, 3), np.uint8)
    src = np.array([[100,40],[210,40],[0,239],[width,239]],np.float32)
    dst = np.array([[0,0],[Width,0],[int(Width/5),Height],[int(Width/5*4),Height]],np.float32)

    M = cv2.getPerspectiveTransform(src, dst)
    warp = cv2.warpPerspective(img1.copy(), M, (Width, Height))

    return warp
    #cv2.imshow('transform', warp)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    #cv2.imwrite("convert.png",warp)




cap = cv2.VideoCapture("test_movie.m4v")
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v') 
convert_out = cv2.VideoWriter('output2.m4v',fourcc, fps, (Width,Height))

before=160
after=0

while(cap.isOpened()):

    #img1 = cv2.imread("frame.png", cv2.IMREAD_COLOR)
    # cv2.imshow('transform', img1)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    ret,img1= cap.read()
    if ret==False:
        break
    out=chokan(img1)
    out  = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    ret2, otsu = cv2.threshold(out, 0, 255, cv2.THRESH_OTSU)
    if abs(ret2-before)>50:
        ret,  out= cv2.threshold(out, before, 255, cv2.THRESH_BINARY)
    else:
        before=ret2
        out=otsu
    #print(ret2)

    out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    out= cv2.line(out,(int(Width/4),int(Height/2)),(int(Width/4),Height),(255,0, 0),5)
    out= cv2.line(out,(0,int(Height/4*3)),(Width,int(Height/4*3)),(255,0, 0),5)
    
    out= cv2.line(out,(int(Width/4*3),0),(int(Width/4*3),Height),(0,255, 0),5)
    out= cv2.line(out,(0,int(Height/2)),(Width,int(Height/2)),(0,255, 0),5)


    convert_out.write(out)
convert_out.release()