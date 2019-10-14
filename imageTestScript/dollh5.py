import os,sys
import time
# from copy import deepcopy

#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

Height = int(320/4)
Width = int(480/4)
#Class_label = ['T', 'curve', 'cross', 'oudan', 'strate', 'stopLine']
Class_label = ['in', 'None']

Class_num = len(Class_label)

def model():
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

def trimming(img):
    h, w, _ = img.shape
    return img[int(h/3):,int(w/6):,]
    # return img[:int(h/2),int(w/4):int(w/2)+int(w/4),]
    # return img

def test(h5,imgPath):
    net = model()
    net.load_weights(h5)
    print("weights complete")
    cap = cv2.VideoCapture(imgPath)
    # print("capopen")

    cnt = 0
    while(cap.isOpened()):
        cnt += 1
        _, img = cap.read()
        if img is None: break
        # print("capread")

        img = trimming(img)

        aap = img
        # aap = deepcopy(img)
        img = img.astype(np.float32)
        img = cv2.resize(img, (Width, Height))

        img = img[:, :, (2,1,0)]
        img = img[np.newaxis, :]
        img = img / 255.
        x = np.array(img, dtype=np.float32)

        pred = net.predict(x, batch_size=1, verbose=0)[0]
        pred_scores = [np.sort(pred)[::-1][0],np.sort(pred)[::-1][1]]
        pred_labels = [Class_label[np.argsort(pred)[::-1][0]],Class_label[np.argsort(pred)[::-1][1]]]

        print("1. {:8d} : {:8.3f}% : {} ".format(cnt,pred_scores[0]*100.0,pred_labels[0]),end="\t||\t")
        print("2. {:8.3f}% : {} ".format(pred_scores[1]*100.0,pred_labels[1]))
        # print("3. {:8.3f}% : {}".format(pred_scores[2]*100.0,pred_labels[2]))

        time.sleep(0.05)

        cv2.imshow('frame',aap)
        cv2.waitKey(1)


    cv2.destroyAllWindows()
    cap.release()

test(sys.argv[1],sys.argv[2])
