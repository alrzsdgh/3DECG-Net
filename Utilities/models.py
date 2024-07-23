# %% [code]
# %% [code]
import numpy as np
import tensorflow as tf


## 3DECG-Net
def SE(inputs, ratio = 8):
    c = np.shape(inputs)[-1]    
    x = tf.keras.layers.GlobalAveragePooling3D()(inputs)   
    x = tf.keras.layers.Dense(c//ratio, activation='relu', use_bias=False)(x)
    x = tf.keras.layers.Dense(c, activation = 'sigmoid', use_bias = False)(x)    
    x = tf.keras.layers.Multiply()([inputs, x])   
    return x


def attBlock(input_tensor):
    k = tf.keras.layers.Dense(1)(input_tensor)
    q = tf.keras.layers.Dense(1)(input_tensor)
    v = tf.keras.layers.Dense(1)(input_tensor)
    alpha = tf.keras.layers.Activation('softmax')(tf.keras.layers.Multiply()([k, q]))
    c = tf.keras.layers.Multiply()([alpha, v])
    return c

def SE_resBlock(input_tensor, n_filters):
    x = tf.keras.layers.BatchNormalization()(input_tensor)
    x = tf.keras.activations.relu(x)
    x = tf.keras.layers.Conv3D(filters=n_filters , kernel_size=(3,3,3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.activations.relu(x)
    x = tf.keras.layers.Conv3D(filters=n_filters ,kernel_size=(3,3,3) , padding='same')(x)
    x = SE(x)
    output_tensor = x + tf.keras.layers.Conv3D(filters=n_filters , kernel_size=(1,1,1))(input_tensor)
    model = tf.keras.models.Model(input_tensor, output_tensor)
    return model, output_tensor

def model_arch():
    inp = tf.keras.Input((45,45,45,1))
    x = tf.keras.layers.ConvLSTM2D(filters=16 , kernel_size=(3,3), padding='same',return_sequences=True)(inp)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.activations.relu(x)
    x = tf.keras.layers.ConvLSTM2D(filters=16 , kernel_size=(3,3), padding='same',return_sequences=True)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.activations.relu(x)
    m, o = SE_resBlock(x, 32)
    m, o = SE_resBlock(o, 32)
    m, o = SE_resBlock(o, 32)
    x = tf.keras.layers.MaxPooling3D((7,7,7))(o)
    x1 = attBlock(x)
    x2 = attBlock(x)
    x3 = attBlock(x)
    x = tf.keras.layers.Concatenate()([x1, x2, x3])
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation = 'relu')(x)
    x = tf.keras.layers.Dense(16, activation = 'relu')(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    x = tf.keras.layers.Dense(7, activation = 'sigmoid')(x)
    return tf.keras.models.Model(inp, x)
##_________________________________________________________________________
## Sobahi et al. model
def attention_based_CNN():
    inp = tf.keras.Input((45,45,45,1))
    x = tf.keras.layers.Conv3D(8, (3,3,3))(inp)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.activations.relu(x)
    x1 = tf.keras.layers.Dropout(0.25)(x)
    x2 = tf.keras.layers.Conv3D(8, (3,3,3), padding='same')(x1)
    x = x2 + x
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.activations.relu(x)
    x3 = tf.keras.layers.Dropout(0.25)(x)
    x4 = tf.keras.layers.Conv3D(8, (3,3,3), padding='same')(x3)
    x5 = x4 + x
    x6 = tf.keras.activations.sigmoid(x5)
    x = tf.math.multiply(x4, x6)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.activations.relu(x)
    x = tf.keras.layers.Dense(7, activation= 'sigmoid')(x)
    return tf.keras.models.Model(inp, x)
##_________________________________________________________________________
# zhang et al. model
def stem(inp_tensor):
    x = tf.keras.layers.Conv3D(32, (3,3,3))(inp_tensor)
    x1 = tf.keras.layers.Conv3D(96, (3,3,3), strides = 2)(x)
    x2 = tf.keras.layers.MaxPooling3D((2,2,2))(x)
    x = tf.keras.layers.Concatenate()([x1, x2])
    x3 = tf.keras.layers.Conv3D(64, (1,1,1), padding = 'same')(x)
    x3 = tf.keras.layers.Conv3D(64, (1,1,7), padding = 'same')(x3)
    x3 = tf.keras.layers.Conv3D(64, (1,7,1), padding = 'same')(x3)
    x3 = tf.keras.layers.Conv3D(96, (3,3,3), padding = 'same')(x3)
    x4 = tf.keras.layers.Conv3D(64, (1,1,1), padding = 'same')(x)
    x4 = tf.keras.layers.Conv3D(96, (3,3,3), padding = 'same')(x4)
    x = tf.keras.layers.Concatenate()([x3, x4])
    x1 = tf.keras.layers.Conv3D(192, (3,3,3), strides = 2)(x)
    x2 = tf.keras.layers.MaxPooling3D((2,2,2))(x)
    x = tf.keras.layers.Concatenate()([x1, x2])
    return tf.keras.models.Model(inp_tensor, x), x

def inresA(inp_tensor):
    x = tf.keras.layers.Conv3D(64, (1,1,1), padding = 'same')(inp_tensor)
    x = tf.keras.layers.Conv3D(64, (3,3,3), padding = 'same')(x)
    x = tf.keras.layers.Conv3D(64, (3,3,3), padding = 'same')(x)
    x1 = tf.keras.layers.Conv3D(64, (1,1,1), padding = 'same')(inp_tensor)
    x1 = tf.keras.layers.Conv3D(64, (3,3,3), padding = 'same')(x1)
    x2 = tf.keras.layers.Conv3D(64, (1,1,1), padding = 'same')(inp_tensor)
    x = tf.keras.layers.Concatenate()([x,x1, x2])
    return tf.keras.models.Model(inp_tensor, x), x

def reducA(inp_tensor):
    x = tf.keras.layers.Conv3D(64, (3,3,3), strides = 2)(inp_tensor)
    x1 = tf.keras.layers.Conv3D(64, (1,1,1), padding = 'same')(inp_tensor)
    x1 = tf.keras.layers.Conv3D(64, (3,3,3), padding = 'same')(x1)
    x1 = tf.keras.layers.Conv3D(64, (3,3,3), strides = 2)(x1)
    x2 = tf.keras.layers.MaxPooling3D((3,3,3), strides=2)(inp_tensor)
    x = tf.keras.layers.Concatenate()([x,x1,x2])
    return tf.keras.models.Model(inp_tensor, x), x

def inresB(inp_tensor):
    x = tf.keras.layers.Conv3D(32, (1,1,1), padding = 'same')(inp_tensor)
    x = tf.keras.layers.Conv3D(32, (1,1,7), padding = 'same')(x)
    x = tf.keras.layers.Conv3D(32, (1,7,1), padding = 'same')(x)
    x1 = tf.keras.layers.Conv3D(32, (1,1,1), padding = 'same')(inp_tensor)
    x = tf.keras.layers.Concatenate()([x,x1])
    return tf.keras.models.Model(inp_tensor, x), x

def reducB(inp_tensor):
    x = tf.keras.layers.Conv3D(32, (1,1,1), padding = 'same')(inp_tensor)
    x = tf.keras.layers.Conv3D(32, (1,1,7), padding = 'same')(x)
    x = tf.keras.layers.Conv3D(32, (1,7,1), padding = 'same')(x)
    x = tf.keras.layers.Conv3D(32, (3,3,3), strides = 2)(x)
    x1 = tf.keras.layers.Conv3D(32, (1,1,1), padding = 'same')(inp_tensor)
    x1 = tf.keras.layers.Conv3D(32, (3,3,3), strides = 2)(x1)
    x2 = tf.keras.layers.MaxPooling3D((3,3,3), strides=2)(inp_tensor)
    x = tf.keras.layers.Concatenate()([x,x1,x2])
    return tf.keras.models.Model(inp_tensor, x), x

def inresC(inp_tensor):
    x = tf.keras.layers.Conv3D(32, (1,1,1), padding = 'same')(inp_tensor)
    x = tf.keras.layers.Conv3D(32, (1,1,3), padding = 'same')(x)
    x = tf.keras.layers.Conv3D(32, (1,3,1), padding = 'same')(x)
    x1 = tf.keras.layers.Conv3D(32, (1,1,1), padding = 'same')(inp_tensor)
    x = tf.keras.layers.Concatenate()([x,x1])
    return tf.keras.models.Model(inp_tensor, x), x

def head(inp_tensor):
    x = tf.keras.layers.BatchNormalization()(inp_tensor)
    x = tf.keras.layers.GlobalAveragePooling3D()(x)
    x = tf.keras.layers.Dense(32)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(7, activation='sigmoid')(x)
    return tf.keras.models.Model(inp_tensor, x), x

def incepres_model():
    inp = tf.keras.Input((45,45,45,1))
    _, a1 = stem(inp)
    _, a2 = inresA(a1)
    _, a3 = reducA(a2)
    _, a4 = inresB(a3)
    _, a5 = reducB(a4)
    _, a6 = inresC(a5)
    _, a7 = head(a6)
    return tf.keras.models.Model(inp, a7)

##-------------------------------------------------------------------------
## Ribeiro et al.

from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, Dropout, BatchNormalization, Activation, Add, Flatten, Dense)
from tensorflow.keras.models import Model

class ResidualUnit(object):
    """Residual unit block (unidimensional).
    Parameters
    ----------
    n_samples_out: int
        Number of output samples.
    n_filters_out: int
        Number of output filters.
    kernel_initializer: str, optional
        Initializer for the weights matrices. See Keras initializers. By default it uses
        'he_normal'.
    dropout_keep_prob: float [0, 1), optional
        Dropout rate used in all Dropout layers. Default is 0.8
    kernel_size: int, optional
        Kernel size for convolutional layers. Default is 17.
    preactivation: bool, optional
        When preactivation is true use full preactivation architecture proposed
        in [1]. Otherwise, use architecture proposed in the original ResNet
        paper [2]. By default it is true.
    postactivation_bn: bool, optional
        Defines if you use batch normalization before or after the activation layer (there
        seems to be some advantages in some cases:
        https://github.com/ducha-aiki/caffenet-benchmark/blob/master/batchnorm.md).
        If true, the batch normalization is used before the activation
        function, otherwise the activation comes first, as it is usually done.
        By default it is false.
    activation_function: string, optional
        Keras activation function to be used. By default 'relu'.
    References
    ----------
    .. [1] K. He, X. Zhang, S. Ren, and J. Sun, "Identity Mappings in Deep Residual Networks,"
           arXiv:1603.05027 [cs], Mar. 2016. https://arxiv.org/pdf/1603.05027.pdf.
    .. [2] K. He, X. Zhang, S. Ren, and J. Sun, "Deep Residual Learning for Image Recognition," in 2016 IEEE Conference
           on Computer Vision and Pattern Recognition (CVPR), 2016, pp. 770-778. https://arxiv.org/pdf/1512.03385.pdf
    """

    def __init__(self, n_samples_out, n_filters_out, kernel_initializer='he_normal',
                 dropout_keep_prob=0.8, kernel_size=17, preactivation=True,
                 postactivation_bn=False, activation_function='relu'):
        self.n_samples_out = n_samples_out
        self.n_filters_out = n_filters_out
        self.kernel_initializer = kernel_initializer
        self.dropout_rate = 1 - dropout_keep_prob
        self.kernel_size = kernel_size
        self.preactivation = preactivation
        self.postactivation_bn = postactivation_bn
        self.activation_function = activation_function

    def _skip_connection(self, y, downsample, n_filters_in):
        """Implement skip connection."""
        # Deal with downsampling
        if downsample > 1:
            y = MaxPooling1D(downsample, strides=downsample, padding='same')(y)
        elif downsample == 1:
            y = y
        else:
            raise ValueError("Number of samples should always decrease.")
        # Deal with n_filters dimension increase
        if n_filters_in != self.n_filters_out:
            # This is one of the two alternatives presented in ResNet paper
            # Other option is to just fill the matrix with zeros.
            y = Conv1D(self.n_filters_out, 1, padding='same',
                       use_bias=False, kernel_initializer=self.kernel_initializer)(y)
        return y

    def _batch_norm_plus_activation(self, x):
        if self.postactivation_bn:
            x = Activation(self.activation_function)(x)
            x = BatchNormalization(center=False, scale=False)(x)
        else:
            x = BatchNormalization()(x)
            x = Activation(self.activation_function)(x)
        return x

    def __call__(self, inputs):
        """Residual unit."""
        x, y = inputs
        n_samples_in = y.shape[1]
        downsample = n_samples_in // self.n_samples_out
        n_filters_in = y.shape[2]
        y = self._skip_connection(y, downsample, n_filters_in)
        # 1st layer
        x = Conv1D(self.n_filters_out, self.kernel_size, padding='same',
                   use_bias=False, kernel_initializer=self.kernel_initializer)(x)
        x = self._batch_norm_plus_activation(x)
        if self.dropout_rate > 0:
            x = Dropout(self.dropout_rate)(x)

        # 2nd layer
        x = Conv1D(self.n_filters_out, self.kernel_size, strides=downsample,
                   padding='same', use_bias=False,
                   kernel_initializer=self.kernel_initializer)(x)
        if self.preactivation:
            x = Add()([x, y])  # Sum skip connection and main connection
            y = x
            x = self._batch_norm_plus_activation(x)
            if self.dropout_rate > 0:
                x = Dropout(self.dropout_rate)(x)
        else:
            x = BatchNormalization()(x)
            x = Add()([x, y])  # Sum skip connection and main connection
            x = Activation(self.activation_function)(x)
            if self.dropout_rate > 0:
                x = Dropout(self.dropout_rate)(x)
            y = x
        return [x, y]


def ribeiro2(last_layer='sigmoid'):
    kernel_size = 16
    kernel_initializer = 'he_normal'
    signal = Input(shape=(100, 12), dtype=np.float32, name='signal')
    x = signal
    x = Conv1D(64, kernel_size, padding='same', use_bias=False,
               kernel_initializer=kernel_initializer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x, y = ResidualUnit(64, 128, kernel_size=kernel_size,
                        kernel_initializer=kernel_initializer)([x, x])
    x, y = ResidualUnit(64, 196, kernel_size=kernel_size,
                        kernel_initializer=kernel_initializer)([x, y])
    x, y = ResidualUnit(32, 256, kernel_size=kernel_size,
                        kernel_initializer=kernel_initializer)([x, y])
    x, _ = ResidualUnit(16, 320, kernel_size=kernel_size,
                        kernel_initializer=kernel_initializer)([x, y])
    x = Flatten()(x)
    diagn = Dense(7, activation=last_layer, kernel_initializer=kernel_initializer)(x)
    model = Model(signal, diagn)
    return model




##_________________________________________________________________________
## Compiling
def CompModel(model):
    model.compile(optimizer=tf.keras.optimizers.Adam() , 
                loss=tf.keras.losses.BinaryCrossentropy() , metrics=['acc'])
    lr_sch = tf.keras.callbacks.LearningRateScheduler(
        lambda epochs: 1e-3 * 10 ** (-epochs/100.0))
    return model, lr_sch