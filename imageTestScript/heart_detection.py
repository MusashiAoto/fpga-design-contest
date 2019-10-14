import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import UInt16
from time import sleep
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import matplotlib.pyplot as plt
import numpy as np

preVel=0.0

pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

def callback(message):
    vel = Twist()
    if message.data < 900:
        vel.linear.x = 0.0
        vel.linear.y = 0.0
        pub.publish(vel)
    else:
        vel.linear.x = preVel
        vel.linear.y = preVel
        pub.publish(vel)
def control():
    preVel=0.0
    rospy.init_node('vel_publisher')
    rospy.Subscriber('/light', UInt16, callback)
    
    
    fstate="stop"
    nstate="stop"
    
    Width=160
    height=120
    XLpoint=Width//6*2
    XRpoint=Width//6*4
    vel = Twist()
    cnt=0
    curcount=0
    while (1):
        cnt+=1
        print(cnt)
        cap = cv2.VideoCapture(1)
        #cap.set(cv2.CAP_PROP_FPS, 120)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, Width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        ret, frame = cap.read()
        #not rospy.is_shutdown()
        cap.release()
        aap=frame
       
        target_pointL=0
        target_pointR=0
        
        print(ret)
        frame = cv2.bitwise_not(frame)
        frame = cv2.GaussianBlur(frame,(3,3),0)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sub,edge_lap=cv2.threshold(frame, 155, 255, cv2.THRESH_BINARY)
        cv2.imwrite("FPGA/HEART/time.png",edge_lap)

        for Y in range(height//4,height//4*3):
                Lcolor=edge_lap[Y,XLpoint]
                Rcolor=edge_lap[Y,XRpoint]
                #print(color)
                if Lcolor==0:
                    target_pointL=Y
                if Rcolor==0:
                    target_pointR=Y

        
      
        #cmode ='false'
       

        if target_pointL-target_pointR>30:
            nstate="curve"
            vel.linear.x =-90 #L120
            vel.linear.y =0 #R
            #pub.publish(vel)
            print("curve")
            #cv2.imwrite("test.yuv",frame)
            curcount+=1
            pub.publish(vel)
            #sleep(0.2)
        elif curcount>2:
            vel.linear.x =-90 #L120
            vel.linear.y =0 #R
            pub.publish(vel)
            print("curve")
            curcount=0

        else :
            nstate="st"
            vel.linear.x =-149
            vel.linear.y =-150
            pub.publish(vel)
            print("Stlaight")
            
            
        if fstate!=nstate:
            
            fstate=nstate
            #print(fstate)
            #pub.publish(vel)
        print("end")
        
        #sleep(0.2)
def strate(vel):
    preVel=1.0
    vel.linear.x=1.0
    vel.linear.y=1.0
    pub.publish(vel)




def curve(vel,of):
    after='c'
    sleep(0.3)
    al=of*0.1
    print(1.0+al)
    prevVel = 1.0
    vel.linear.x = -1.2
    vel.linear.y = -2.4
    print("curve")
    pub.publish(vel)
    #sleep(4)
    before='c'

def kaihi(vel,preVel):
    prevVel = 1.0
    vel.linear.x = 0.1
    vel.linear.y = 2.0
    pub.publish(vel)
    sleep(0.8)
    vel.linear.x = 1.0
    vel.linear.y = 1.0
    pub.publish(vel)
    sleep(2)
    vel.linear.x = 2.0
    vel.linear.y = 0.1
    pub.publish(vel)
    sleep(0.8)
    vel.linear.x = 1.0
    vel.linear.y = 1.0
    pub.publish(vel)
    sleep(2)
    vel.linear.x = 2.0
    vel.linear.y = 0.1
    pub.publish(vel)
    sleep(0.8)
    vel.linear.x = 1.0
    vel.linear.y = 1.0
    pub.publish(vel)
    sleep(1.5)
    vel.linear.x = 0.1
    vel.linear.y = 2.0
    pub.publish(vel)
    sleep(0.8)
    vel.linear.x = 1.0
    vel.linear.y = 1.0



if __name__ == '__main__':
    try:
        control()
        #callcpiFPGA()
    except rospy.ROSInterruptException:
        pass

