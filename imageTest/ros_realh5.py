import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import UInt16
from time import sleep
import os,sys
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





os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
Height = int(320/4)
Width = int(480/4)
# Height = 160
# Width = 160
Class_label = ['T', 'curve', 'cross', 'oudan', 'strate', 'stopLine']
Class_num = len(Class_label)

def vgg16():
    print("from keras.models import Model")
    from keras.models import Model
    print("from keras.layers import Dense, Dropout, Flatten")
    from keras.layers import Dense, Dropout, Flatten
    print("VGG16")
    from keras.applications.vgg16 import VGG16

    model = VGG16(include_top=False,
                  weights=None,
                  input_shape=(Height, Width, 3))

    last = model.output
    x = Flatten()(last)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(Class_num, activation='softmax')(x)
    model = Model(model.input, x)
    print("model complete")
    return model

def model():
    from keras.applications.mobilenet import MobileNet
    from keras.models import Model
    from keras.layers import Dense, Dropout, Flatten
    model = MobileNet(include_top=False,
                  weights=None,
                  input_shape=(Height, Width, 3))

    last = model.output
    x = Flatten()(last)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(Class_num, activation='softmax')(x)
    model = Model(model.input, x)
    print("model complete")
    return model

def readAndTrimming(img):
    aap = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #print("1")
    size = img.shape
    #print("2")
    img = img[int(size[0]/2):-1,:]
    #print("3")
    img = cv2.Canny(img, 105, 110)
    #print("4")
    img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    #print("")
    img = cv2.resize(img, (Width, Height))
    return img

def test(h5):
    net = vgg16()
    net.load_weights(h5)
    print("weights complete")
    #cap = cv2.VideoCapture(imgPath)
    # print("capopen")


    global prevVel
    rospy.init_node('vel_publisher')
    rospy.Subscriber('/light', UInt16, callback)
    cap = cv2.VideoCapture(1)
    XLpoint=40
    XRpoint=80
    cnt=0
    while not rospy.is_shutdown():
        cnt += 1
        _, img = cap.read()
        print("capread")
        img = readAndTrimming(img).astype(np.float32)
        aap = img
        target_pointL=0
        target_pointR=0

        ret,edge_lap=cv2.threshold(aap, 20, 255, cv2.THRESH_BINARY)
        img = cv2.resize(img, (Width, Height))

        img = img[:, :, (2,1,0)]
        img = img[np.newaxis, :]
        img = img / 255.
        x = np.array(img, dtype=np.float32)

        pred = net.predict(x, batch_size=1, verbose=0)[0]
        pred_scores = [np.sort(pred)[::-1][0],np.sort(pred)[::-1][1],np.sort(pred)[::-1][2]]
        pred_labels = [Class_label[np.argsort(pred)[::-1][0]],Class_label[np.argsort(pred)[::-1][1]],Class_label[np.argsort(pred)[::-1][2]]]

        print("1. {:8d} : {:8.3f}% : {} ".format(cnt,pred_scores[0]*100.0,pred_labels[0]),end="\t||\t")
        #print("2. {:8.3f}% : {} ".format(pred_scores[1]*100.0,pred_labels[1]),end="\t||\t")
        #print("3. {:8.3f}% : {}".format(pred_scores[2]*100.0,pred_labels[2]))


        vel = Twist()



        cmode ='false'

        if pred_labels[0]=="strate":

            vel.linear.x =-0.5
            vel.linear.y =-0.5
            pub.publish(vel)
        else :

            print("else")
            vel.linear.x =-0.0
            vel.linear.y =-0.0
            pub.publish(vel)
        #cv2.imshow('frame',aap)
        #cv2.waitKey(1)


    #cv2.destroyAllWindows()
    cap.release()

test(sys.argv[1])
