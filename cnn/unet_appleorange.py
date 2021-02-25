import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input, Flatten, Dense, Conv2D, BatchNormalization, Dropout, Activation, UpSampling2D, Concatenate, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization 




gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:

    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:

    print(e)



IMAGE_SIZE = 128
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


f1 = Flatten()(g_output)
f2 = Dense(512)(f1)
f3 = Dropout(rate = 0.4)(f2)
f4 = BatchNormalization()(f3)
f5 = LeakyReLU()(f4)
f6 = Dense(1)(f5)
f7 = Activation('sigmoid')(f6)


g_model = Model(img_1, f7)

g_model.summary()

opt = Adam(lr=0.0005)
g_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])




from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_directory(
        './apple2orange/train/', 
        target_size=(128, 128),  
        batch_size=32,
        class_mode='binary')


validation_generator = validation_datagen.flow_from_directory(
        './apple2orange/test/', 
        target_size=(128, 128), 
        batch_size=32,
        class_mode='binary')

history = g_model.fit(
      train_generator,
      steps_per_epoch=10,  
      epochs=15,
      verbose=1,
      validation_data = validation_generator,
      validation_steps=8)