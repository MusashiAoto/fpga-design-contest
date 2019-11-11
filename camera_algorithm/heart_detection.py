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

#shikiichi yuugata white_balance_temperature=2800 exposure_absolute=180->50
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
    Lcount=0
    motorcurve=-80
    #motorforward=-120
    while (1):
        cnt+=1
        #print(cnt)
        cap = cv2.VideoCapture(1)
        #cap.set(cv2.CAP_PROP_FPS, 120)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, Width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        ret, frame = cap.read()
        #not rospy.is_shutdown()
        cap.release()
        aap=frame
        cv2.imwrite("FPGA/HEART/aap.png",aap)
        target_pointL=0
        target_pointR=0
        
        print(ret)
        #frame = cv2.bitwise_not(frame)
        frame = cv2.GaussianBlur(frame,(3,3),0)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        #sub,edge_lap=cv2.threshold(frame, 150, 255, cv2.THRESH_BINARY_INV)
       

        edge_lap=black(aap)
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
            vel.linear.x =motorcurve #L120
            vel.linear.y =0 #R
            #pub.publish(vel)
            print("Rcurve")
            #cv2.imwrite("test.yuv",frame)
            curcount+=1
            pub.publish(vel)
            #sleep(0.2)

        elif target_pointR-target_pointL>40 and target_pointR-target_pointL<60:
            vel.linear.x =motorcurve #L120 dounikasitai
            vel.linear.y = 0#R
            print("store_curve")
            pub.publish(vel)

        elif curcount>2:
            vel.linear.x =-150 #L120
            vel.linear.y =0 #R
            pub.publish(vel)
            print("more..curve")
            curcount=0
        elif target_pointR-target_pointL>35:
            if Lcount<1:
                vel.linear.x =motorcurve #L120 gyakukamo
                vel.linear.y =0 #R
                print("Ldakedocurve")
                pub.publish(vel)
                Lcount+=1
            else :
                Lcount-=1
                vel.linear.x =0 #L120 gyakukamo
                vel.linear.y =motorcurve #R
                pub.publish(vel)

            print("Lcurve")
            
            

        else :
            nstate="st"
            vel.linear.x =-130
            vel.linear.y =-130
            pub.publish(vel)
            print("Stlaight")
            
            
        if fstate!=nstate:
            
            fstate=nstate
            #print(fstate)
            #pub.publish(vel)
        #print("end")
        
        #sleep(0.2)
def black(frame):
    ave=33
    frame = cv2.GaussianBlur(frame,(3,3),0)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kuro=np.amin(frame)
    print(kuro)
    bright=kuro-ave

    print(bright)

    fractal_controll=0
    if bright>=0:
        fractal_controll=bright//10
        if fractal_controll>5:
            fractal_controll=4.5
        if fractal_controll==0:
            fractal_controll=1+bright*0.1
    else:
        fractal_controll=bright*2
        

    print(fractal_controll)

    #frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret,edge_lap=cv2.threshold(frame, 55*fractal_controll, 255, cv2.THRESH_BINARY_INV)

    return edge_lap



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

