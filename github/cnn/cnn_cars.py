import tensorflow as tf

import numpy as np

from tensorflow.keras.layers import Input, Flatten, Dense, Conv2D, BatchNormalization, LeakyReLU, Dropout, Activation, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import tensorflow.keras.backend as K 

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


input_layer = Input((150,150,3))

x = Conv2D(filters = 16, kernel_size = 3, padding = 'same')(input_layer)
x = BatchNormalization()(x)
x = LeakyReLU()(x)
x = MaxPooling2D(pool_size=2)(x)

x = Conv2D(filters = 32, kernel_size = 3, padding = 'same')(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)
x = MaxPooling2D(pool_size=2)(x)
x = Dropout(rate = 0.4)(x)

x = Flatten()(x)

x = Dense(512)(x)
x = Dropout(rate = 0.4)(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)

x = Dense(1)(x)
output_layer = Activation('sigmoid')(x)

model = Model(input_layer, output_layer)

model.summary()

opt = Adam(lr=0.0005)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])


from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)

# Flow training images in batches of 128 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
        './cars/cars_train/',  # This is the source directory for training images
        target_size=(150, 150),  
        batch_size=32,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

# Flow training images in batches of 128 using train_datagen generator
validation_generator = validation_datagen.flow_from_directory(
        './cars/cars_test/',  # This is the source directory for training images
        target_size=(150, 150), 
        batch_size=32,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

history = model.fit(
      train_generator,
      steps_per_epoch=8,  
      epochs=10,
      verbose=1,
      validation_data = validation_generator,
      validation_steps=8)