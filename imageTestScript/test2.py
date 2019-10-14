import os,sys,cv2
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
Height = int(320/4)
Width = int(480/4)
Class_label = ['T', 'curve', 'cross', 'oudan', 'strate', 'stopLine']
Class_num = len(Class_label)

def show(img):
    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def model():
    from keras.models import Sequential, Model
    from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Activation, Flatten
    from keras.applications.vgg16 import VGG16
    model = VGG16(include_top=False,
                  input_shape=(Height, Width, 3))

    last = model.output
    x = Flatten()(last)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(Class_num, activation='softmax')(x)
    model = Model(model.input, x)
    return model

def readAndTrimming(path):
    img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    size = img.shape
    img = img[int(size[0]/2):-1,:]

    img = cv2.Canny(img, 105, 110)
    img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    img = cv2.resize(img, (Width, Height))
    return img

def test(h5,imgPath):
    net = model()
    net.load_weights(h5)

    img = readAndTrimming(imgPath).astype(np.float32)

    img = img[:, :, (2,1,0)]
    img = img[np.newaxis, :]
    img = img / 255.
    x = np.array(img, dtype=np.float32)

    pred = net.predict(x, batch_size=1, verbose=0)[0]
    pred_scores = [np.sort(pred)[::-1][0],np.sort(pred)[::-1][1],np.sort(pred)[::-1][2]]
    pred_labels = [Class_label[np.argsort(pred)[::-1][0]],Class_label[np.argsort(pred)[::-1][1]],Class_label[np.argsort(pred)[::-1][2]]]

    print("1. {} : {:8.3f}% : {} ".format(os.path.basename(imgPath),pred_scores[0]*100.0,pred_labels[0]),end="\t||\t")
    print("2. {:8.3f}% : {} ".format(pred_scores[1]*100.0,pred_labels[1]),end="\t||\t")
    print("3. {:8.3f}% : {}".format(pred_scores[2]*100.0,pred_labels[2]))

if __name__ == "__main__":
	test(sys.argv[1],sys.argv[2])
