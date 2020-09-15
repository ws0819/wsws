# 미술관에 GAN 딥러닝 실전 프로젝트
# 뉴럴 스타일 트랜스퍼
# 책에 있는 코드가 안되서 약간 

from keras.preprocessing.image import load_img, img_to_array, save_img
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Restrict TensorFlow to only use the fourth GPU
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')

        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)



target_image_path = './data4/tubingen.jpg'

style_reference_image_path = './data4/starry-night.jpg'


width, height = load_img(target_image_path).size
img_height = 600
img_width = int(width * img_height / height)

print(img_width, img_height)

import numpy as np
from keras.applications import vgg19

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(img_height, img_width))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return img

def deprocess_image(x):
    if K.image_data_format() == 'channels_first':
        x = x.reshape((3, img_nrows, img_ncols))
        x = x.transpose((1, 2, 0))
    else:
        x = x.reshape((img_height, img_width, 3))
    
    x[:, :, 0] += 103.3939
    x[:, :, 1] += 116.6779
    x[:, :, 2] += 123.368
   
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x

from keras import backend as K

target_image = K.constant(preprocess_image(target_image_path))
style_reference_image = K.constant(preprocess_image(style_reference_image_path))


combination_image = K.placeholder((1, img_height, img_width, 3))


input_tensor = K.concatenate([target_image, style_reference_image, combination_image], axis=0)


model = vgg19.VGG19(input_tensor=input_tensor, weights='imagenet', include_top=False)

def content_loss(base, combination):
    return K.sum(K.square(combination - base))

def gram_matrix(x):
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram

def style_loss(style, combination):
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = img_height * img_width
    return K.sum(K.square(S - C)) / (4. * (channels ** 2) * (size ** 2))

def total_variation_loss(x):
    a = K.square(
        x[:, :img_height - 1, :img_width - 1, :] - x[:, 1:, :img_width - 1, :])
    b = K.square(
        x[:, :img_height - 1, :img_width - 1, :] - x[:, :img_height - 1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))



 
outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])

content_layer = 'block5_conv2'

style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1']

total_variation_weight = 20
style_weight = 100
content_weight = 1


loss = K.variable(0.0)
layer_features = outputs_dict['block5_conv2']
target_image_features = layer_features[0, :, :, :]
combination_features = layer_features[2, :, :, :]
loss = loss + content_weight * content_loss(target_image_features, combination_features)
for layer_name in style_layers:
    layer_features = outputs_dict[layer_name]
    style_reference_features = layer_features[1, :, :, :]
    combination_features = layer_features[2, :, :, :]
    sl = style_loss(style_reference_features, combination_features)
    loss = loss + (style_weight / len(style_layers)) * sl
loss = loss + total_variation_weight * total_variation_loss(combination_image)


grads = K.gradients(loss, combination_image)



outputs = [loss]
if isinstance(grads, (list, tuple)):
    outputs += grads
else:
    outputs.append(grads)

f_outputs = K.function([combination_image], outputs)

def eval_loss_and_grads(x):
    if K.image_data_format() == 'channels_first':
        x = x.reshape((1, 3, img_height, img_width))
    else:
        x = x.reshape((1, img_height, img_width, 3))
    outs = f_outputs([x])
    loss_value = outs[0]
    if len(outs[1:]) == 1:
        grad_values = outs[1].flatten().astype('float64')
    else:
        grad_values = np.array(outs[1:]).flatten().astype('float64')
    return loss_value, grad_values

class Evaluator(object):

    def __init__(self):
        self.loss_value = None
        self.grads_values = None

    def loss(self, x):
        assert self.loss_value is None
        loss_value, grad_values = eval_loss_and_grads(x)
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values


evaluator = Evaluator()

from scipy.optimize import fmin_l_bfgs_b

result_file = './data4/style_transfer_result.jpg'
iterations = 100


x = preprocess_image(target_image_path)
x = x.flatten()
for i in range(iterations):
    x, min_val, info = fmin_l_bfgs_b(
        evaluator.loss
        , x
        , fprime=evaluator.grads
        , maxfun=20
        )

    if i % 100 == 0:
        print('.', end=' ')
        print('손실 값:', min_val)


img = x.copy().reshape((img_height, img_width, 3))
img = deprocess_image(img)

save_img(result_file, img)

from matplotlib import pyplot as plt


plt.imshow(load_img(target_image_path, target_size=(img_height, img_width)))
plt.figure()


plt.imshow(load_img(style_reference_image_path, target_size=(img_height, img_width)))
plt.figure()


plt.imshow(img)
plt.show()

