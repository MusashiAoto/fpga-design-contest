import os,sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
capture = cv2.VideoCapture(1)
Width = int(capture.get(3))
Height = int(capture.get(4))
print(Width,Height)
capture.set(cv2.CAP_PROP_FPS, 30)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 160)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 120)
ret, frame = capture.read()
aap=frame
frame = cv2.bitwise_not(frame)
frame = cv2.GaussianBlur(frame,(3,3),0)
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
ret,edge_lap=cv2.threshold(frame, 90, 255, cv2.THRESH_BINARY)
cv2.imwrite("usb.png",edge_lap)
cv2.imwrite("tete.png",aap)
capture.release()
