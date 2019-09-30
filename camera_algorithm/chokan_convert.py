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


def changecontrast(frame,contraststate):
    ave=33
    frame = cv2.GaussianBlur(frame,(5,5),0)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img=frame
    size = frame.shape
    #img= img[int(size[0]/4*3):-1,int(size[1]/4):int(size[1]/4)*3]
    #cv2.imshow('frame',img)
    kuro=np.amin(img)
    #print(kuro)

    #avecontrast=0


    
    bright=kuro-ave

    #print(bright)

    fractal_controll=kuro*(1.2+(0.006*bright))#1.55
    if fractal_controll>200:
        fractal_controll=210
    elif fractal_controll<70:
        fractal_controll=70


    if contraststate-fractal_controll>13:
        fractal_controll=contraststate
    contraststate=fractal_controll
    #if bright>=0:
     #   fractal_controll=bright//10
      #  if fractal_controll>5:
       #     fractal_controll=4.5
        #if fractal_controll==0:
         #   fractal_controll=1+bright*0.1
    #else:
     #   fractal_controll=bright*2


    print(fractal_controll)
    return fractal_controll,frame,contraststate

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
    #cv2.imshow('frame',img1)
    frame=img1
    fractal_controll,frame,contraststate= changecontrast(frame,contraststate)
    ret,edge_lap=cv2.threshold(frame, 150, 255, cv2.THRESH_BINARY)
    #cv2.imshow('frame',frame)
    out=chokan(edge_lap)
    out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    convert_out.write(out)
    #time.sleep(0.05)
convert_out.release()


# img1 = cv.imread("timeout.png", cv.IMREAD_COLOR)
# # cv.imshow('transform', img1)
# # cv.waitKey(0)
# # cv.destroyAllWindows()
# height, width, channels = img1.shape[:3]
# image = np.zeros((700, 700, 3), np.uint8)
# src = np.array([[40,16],[107,16],[0,119],[width,119]],np.float32)
# dst = np.array([[0,0],[50,0],[int(50/5),70],[int(50/5*4),70]],np.float32)

# M = cv.getPerspectiveTransform(src, dst)
# warp = cv.warpPerspective(img1.copy(), M, (50, 70))
# cv.imshow('transform', warp)
# cv.waitKey(0)
# cv.destroyAllWindows()