import os,sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2


savePath = "testImage"
capture = cv2.VideoCapture(1)
capture.set(cv2.CAP_PROP_FPS, 30)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 160)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 120)

if os.path.isdir(savePath):
    for file in os.listdir(savePath):
        if not os.path.isdir:os.remove(file)

os.mkdir(savePath)

for i in range(10):
    autoFlag = 0
    th = i*20
    os.system('v4l2-ctl -d /dev/video1 -c exposure_auto={} -c exposure_absolute={}'.format(autoFlag,th))
    ret, frame = capture.read()
    cv2.imwrite("testImage/auto-{}_absolute-{}.png".format(autoFlag,th),frame)

capture.release()
