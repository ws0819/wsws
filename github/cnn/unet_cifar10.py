import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input, Flatten, Dense, Conv2D, BatchNormalization, LeakyReLU, Dropout, Activation, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

#--------------------------------------------

from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization 
#몰라 어케든 우겨넣음 자세한건 메모장에 적어둘께여
from tensorflow.keras.layers import BatchNormalization, Activation, LeakyReLU, UpSampling2D, Conv2D, ReLU
from tensorflow.keras.models import Sequential, Model
from keras.initializers import RandomNormal
from keras.optimizers import Adam
from keras.utils import plot_model
from keras import backend as K

from tensorflow.keras.datasets import cifar10

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)



NUM_CLASSES = 10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
y_train = to_categorical(y_train, NUM_CLASSES)
y_test = to_categorical(y_test, NUM_CLASSES)

IMAGE_SIZE = 32
image_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)
img_1 = Input(shape = image_shape)

 #다운샘플링#
#d1 filters = 32
d1_1 = Conv2D(filters = 32, kernel_size=(4, 4), strides = (2,2), padding = 'same')(img_1)
d1_2 = InstanceNormalization(axis=-1, center=False, scale=False)(d1_1)
d1_3 = Activation('relu')(d1_2)
#d2 filters*2
d2_1 = Conv2D(filters = 64, kernel_size=(4, 4), strides = (2,2), padding = 'same')(d1_3)
d2_2 = InstanceNormalization(axis=-1, center=False, scale=False)(d2_1)
d2_3 = Activation('relu')(d2_2)
#d3 filsters*4
d3_1 = Conv2D(filters = 128 , kernel_size=(4, 4), strides = (2,2), padding = 'same')(d2_3)
d3_2 = InstanceNormalization(axis=-1, center=False, scale=False)(d3_1)
d3_3 = Activation('relu')(d3_2)
#d4 filsters*8
d4_1 = Conv2D(filters = 256 , kernel_size=(4, 4), strides = (2,2), padding = 'same')(d3_3)
d4_2 = InstanceNormalization(axis=-1, center=False, scale=False)(d4_1)
d4_3 = Activation('relu')(d4_2)

 #업샘플링#
#u1 d3연결 filters*4
u1_1 = UpSampling2D(size = 2)(d4_3)
u1_2 = Conv2D(filters = 128, kernel_size=(4,4), strides = (1,1), padding = 'same')(u1_1)
u1_3 = InstanceNormalization(axis=-1, center=False, scale=False)(u1_2)
u1_4 = Activation('relu')(u1_3)
u1_5 = Concatenate()([u1_4, d3_3])
#u2 d2연결 filters*2
u2_1 = UpSampling2D(size = 2)(u1_5)
u2_2 = Conv2D(filters = 64, kernel_size=(4,4), strides = (1,1), padding = 'same')(u2_1)
u2_3 = InstanceNormalization(axis=-1, center=False, scale=False)(u2_2)
u2_4 = Activation('relu')(u2_3)
u2_5 = Concatenate()([u2_4, d2_3])
#u3 d1연결 filters = 32
u3_1 = UpSampling2D(size = 2)(u2_5)
u3_2 = Conv2D(filters = 32, kernel_size=(4,4), strides = (1,1), padding = 'same')(u3_1)
u3_3 = InstanceNormalization(axis=-1, center=False, scale=False)(u3_2)
u3_4 = Activation('relu')(u3_3)
u3_5 = Concatenate()([u3_4, d1_3])
#u4
u4 = UpSampling2D(size = 2)(u3_5)
#ouput
g_output = Conv2D(3, kernel_size=(4,4), strides = (1,1), padding = 'same', activation = 'tanh')(u4)

#마무리?
f1 = Flatten()(g_output)
f2 = Dense(512)(f1)
f3 = Dropout(rate = 0.4)(f2)
f4 = BatchNormalization()(f3)
f5 = LeakyReLU()(f4)
f6 = Dense(10)(f5)
f7 = Activation('sigmoid')(f6)


g_model = Model(img_1, f7)

g_model.summary()

opt = Adam(lr=0.0005)
g_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])



# 모델의 실행


g_model.fit(x_train, y_train, batch_size=32, epochs=10, shuffle=True, validation_data=(x_test, y_test))

g_model.layers[6].get_weights()

g_model.evaluate(x_test, y_test, batch_size=1000)

CLASSES = np.array(['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'])

preds = g_model.predict(x_test)
preds_single = CLASSES[np.argmax(preds, axis = -1)]
actual_single = CLASSES[np.argmax(y_test, axis = -1)]

import matplotlib.pyplot as plt

n_to_show = 10
indices = np.random.choice(range(len(x_test)), n_to_show)

fig = plt.figure(figsize=(15, 3))
fig.subplots_adjust(hspace=0.5, wspace=0.5)

for i, idx in enumerate(indices):
    img = x_test[idx]
    ax = fig.add_subplot(1, n_to_show, i+1)
    ax.axis('off')
    ax.text(0.5, -0.35, 'pred = ' + str(preds_single[idx]), fontsize=10, ha='center', transform=ax.transAxes) 
    ax.text(0.5, -0.7, 'act = ' + str(actual_single[idx]), fontsize=10, ha='center', transform=ax.transAxes)
    ax.imshow(img)