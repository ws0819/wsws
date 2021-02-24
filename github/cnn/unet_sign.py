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

from collections import namedtuple
from skimage.color import rgb2lab
from skimage.transform import resize
from sklearn.model_selection import train_test_split
import glob
import matplotlib.pyplot as plt



Dataset = namedtuple('Dataset', ['X', 'y'])
N_CLASSES = 43
RESIZED_IMAGE = (32, 32)
#이미지 불러오기 함수
def to_tf_format(imgs):
    return np.stack([img[:, :, np.newaxis] for img in imgs], axis=0).astype(np.float32)

def read_dataset_ppm(rootpath, n_labels, resize_to):
  images = []
  labels = []
  
  for c in range(n_labels):
    full_path = rootpath + '/' + format(c, '05d') + '/'
    for img_name in glob.glob(full_path + "*.ppm"):
      
      img = plt.imread(img_name).astype(np.float32)
      img = rgb2lab((img-127.5)/127.5)[:,:,0]
      if resize_to:
        img = resize(img, resize_to, mode='reflect')
      
      label = np.zeros((n_labels, ), dtype=np.float32)
      label[c] = 1.0
      
      images.append(img.astype(np.float32))
      labels.append(label)

  return Dataset(X = to_tf_format(images).astype(np.float32),
                 y = np.matrix(labels).astype(np.float32))

dataset = read_dataset_ppm('GTSRB/Final_Training/Images', N_CLASSES, RESIZED_IMAGE)

idx_train, idx_test = train_test_split(range(dataset.X.shape[0]), test_size=0.25, random_state=101)
X_train = dataset.X[idx_train, :, :, :]
X_test = dataset.X[idx_test, :, :, :]
y_train = dataset.y[idx_train, :]
y_test = dataset.y[idx_test, :]

def minibatcher(X, y, batch_size, shuffle):
  assert X.shape[0] == y.shape[0]
  n_samples = X.shape[0]
  
  if shuffle:
    idx = np.random.permutation(n_samples)
  else:
    idx = list(range(n_samples))
  
  for k in range(int(np.ceil(n_samples/batch_size))):
    from_idx = k*batch_size
    to_idx = (k+1)*batch_size
    yield X[idx[from_idx:to_idx], :, :, :], y[idx[from_idx:to_idx], :]

for mb in minibatcher(X_train, y_train, 10000, True):
    print(mb[0].shape, mb[1].shape)




IMAGE_SIZE = 32
image_shape = (IMAGE_SIZE, IMAGE_SIZE, 1)
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
f6 = Dense(43)(f5)
f7 = Activation('sigmoid')(f6)


g_model = Model(img_1, f7)

g_model.summary()

opt = Adam(lr=0.0005)
g_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

g_model.fit(X_train, y_train, batch_size=32, epochs=10, shuffle=True, validation_data=(X_test, y_test))

g_model.layers[6].get_weights()

g_model.evaluate(X_test, y_test, batch_size=1000)

