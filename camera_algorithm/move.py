import numpy as np
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import time
import cv2
capture = cv2.VideoCapture(1)
#capture2 = cv2.VideoCapture(2)
Width = int(capture2.get(3))
Height = int(capture2.get(4))
Width=640
Height=480
fps=30

print(Width,Height)
capture.set(cv2.CAP_PROP_FPS, fps)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, Width)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, Height)

#capture2.set(cv2.CAP_PROP_FPS, fps)
#capture2.set(cv2.CAP_PROP_FRAME_WIDTH, Width)
#capture2.set(cv2.CAP_PROP_FRAME_HEIGHT, Height)

start=time.time()
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v') 



print(Width,Height)
out = cv2.VideoWriter('output5.m4v',fourcc, fps, (Width,Height))
#out2 = cv2.VideoWriter('output2.m4v',fourcc, fps, (Width,Height))



def cam1(out,capture):

    ret, frame = capture.read()
    if ret==False:
        print("false")
    out.write(frame)



# def cam2(out2,capture2):

#     ret2, frame2 = capture2.read()
#     if ret2==False:
#         print("false2")
#     out2.write(frame2)

flag=0
while(capture.isOpened()):
    est=time.time()

    cam1(out,capture)

    #cam2(out2,capture2)


 
    if est-start>100:
        break


capture.release()

#capture2.release()
out.release()
#out2.release()
