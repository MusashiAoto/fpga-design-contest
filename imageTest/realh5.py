import os,sys,cv2
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
Height = int(320/4)
Width = int(480/4)
Class_label = ['T', 'curve', 'cross', 'oudan', 'strate', 'stopLine']
Class_num = len(Class_label)
from InceptionResNetV2 import model as model

# def model():
#     from keras.models import Model
#     from keras.layers import Dense, Dropout, Flatten
#     from keras.applications.inception_resnet_v2 import InceptionResNetV2
#     model = InceptionResNetV2(include_top=False,
#                                 weights=None,
#                                 input_shape=(Height, Width, 3))


#     last = model.output
#     x = Flatten()(last)
#     x = Dense(4096, activation='relu')(x)
#     x = Dropout(0.5)(x)
#     x = Dense(4096, activation='relu')(x)
#     x = Dropout(0.5)(x)
#     x = Dense(Class_num, activation='softmax')(x)
#     model = Model(model.input, x)
#     return model

    # from keras.models import Model
    # from keras.layers import Dense, Dropout, Flatten
    # from keras.applications.vgg16 import VGG16
    # model = VGG16(include_top=False,
    #               weights=None,
    #               input_shape=(Height, Width, 3))

    # last = model.output
    # x = Flatten()(last)
    # x = Dense(4096, activation='relu')(x)
    # x = Dropout(0.5)(x)
    # x = Dense(4096, activation='relu')(x)
    # x = Dropout(0.5)(x)
    # x = Dense(Class_num, activation='softmax')(x)
    # model = Model(model.input, x)
    # return model

def readAndTrimming(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    size = img.shape
    img = img[int(size[0]/2):-1,:]

    img = cv2.Canny(img, 105, 110)
    img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    img = cv2.resize(img, (Width, Height))
    #print("capopen")
    return img

def test(imgPath):
    net = model()
    print("{}層".format(len(net.layers)))
    #print("capopen")
    net.load_weights("out/20191109150343_e10_CNN.h5",by_name=False)

    
    cap = cv2.VideoCapture(imgPath)
    # print("capopen")

    cnt = 0
    while(cap.isOpened()):
        cnt += 1
        _, img = cap.read()
        print("capread")
        img = readAndTrimming(img).astype(np.float32)
        aap = img
        img = cv2.resize(img, (Width, Height))

        img = img[:, :, (2,1,0)]
        img = img[np.newaxis, :]
        img = img / 255.
        x = np.array(img, dtype=np.float32)

        pred = net.predict(x, batch_size=1, verbose=0)[0]
        pred_scores = [np.sort(pred)[::-1][0],np.sort(pred)[::-1][1],np.sort(pred)[::-1][2]]
        pred_labels = [Class_label[np.argsort(pred)[::-1][0]],Class_label[np.argsort(pred)[::-1][1]],Class_label[np.argsort(pred)[::-1][2]]]

        print("1. {:8d} : {:8.3f}% : {} ".format(cnt,pred_scores[0]*100.0,pred_labels[0]),end="\t||\t")
        print("2. {:8.3f}% : {} ".format(pred_scores[1]*100.0,pred_labels[1]),end="\t||\t")
        print("3. {:8.3f}% : {}".format(pred_scores[2]*100.0,pred_labels[2]))


        cv2.imshow('frame',aap)
        cv2.waitKey(1)


    cv2.destroyAllWindows()
    cap.release()

test(sys.argv[1])
