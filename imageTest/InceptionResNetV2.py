import keras
from keras.models import Sequential, Model
from keras.layers import Conv2D,MaxPooling2D,Dense, Dropout, Activation, Flatten
import config as cf

def model():
    from keras.applications.inception_resnet_v2 import InceptionResNetV2
    model = InceptionResNetV2(include_top=False,
                              weights='imagenet',
                              input_shape=(cf.Height, cf.Width, 3))


    last = model.output
    x = Flatten()(last)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(cf.Class_num, activation='softmax')(x)
    model = Model(model.input, x)
    return model