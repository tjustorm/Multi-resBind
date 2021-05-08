import keras
from keras.layers import Dense, Conv1D, BatchNormalization, Activation
from keras.layers import AveragePooling1D, Input, Flatten
from keras import backend as K
from keras.models import Model
from models.attention_module import attach_attention_module

def ResidualBind(input_shape=(150, 14), num_class=27, classification=True):

        def residual_block(input_layer, filter_size, activation='relu',
                           dilated=False):

            if dilated:
                factor = [2, 4, 8]
            else:
                factor = [1]
            num_filters = input_layer.shape.as_list()[-1]

            nn = keras.layers.Conv1D(filters=num_filters,
                                     kernel_size=filter_size,
                                     activation=None,
                                     use_bias=False,
                                     padding='same',
                                     dilation_rate=1,
                                     )(input_layer)
            nn = keras.layers.BatchNormalization()(nn)
            for f in factor:
                nn = keras.layers.Activation('relu')(nn)
                nn = keras.layers.Dropout(0.1)(nn)
                nn = keras.layers.Conv1D(filters=num_filters,
                                         kernel_size=filter_size,
                                         strides=1,
                                         activation=None,
                                         use_bias=False,
                                         padding='same',
                                         dilation_rate=f,
                                         )(nn)
                nn = keras.layers.BatchNormalization()(nn)
            nn = keras.layers.add([input_layer, nn])
            return keras.layers.Activation(activation)(nn)

        # input layer
        inputs = keras.layers.Input(shape=input_shape)

        # layer 1
        nn = keras.layers.Conv1D(filters=96,
                                 kernel_size=11,
                                 strides=1,
                                 activation=None,
                                 use_bias=False,
                                 padding='same',
                                 )(inputs)
        nn = keras.layers.BatchNormalization()(nn)
        nn = keras.layers.Activation('relu')(nn)
        nn = keras.layers.Dropout(0.1)(nn)

        # dilated residual block
        nn = residual_block(nn, filter_size=3, dilated=True)

        # average pooling
        nn = keras.layers.AveragePooling1D(pool_size=10)(nn)
        nn = keras.layers.Dropout(0.2)(nn)

        """
        # layer 2
        nn = keras.layers.Conv1D(filters=128,
                                 kernel_size=3,
                                 strides=1,
                                 activation=None,
                                 use_bias=False,
                                 padding='same',
                                 )(nn)                               
        nn = keras.layers.BatchNormalization()(nn)
        nn = keras.layers.Activation('relu')(nn)
        nn = keras.layers.Dropout(0.1)(nn)
        nn = residual_block(nn, filter_size=3, dilated=False)

        nn = keras.layers.AveragePooling1D(pool_size=4, 
                                           strides=4, 
                                           )(nn)
        nn = keras.layers.Dropout(0.3)(nn)
        """
        # Fully-connected NN
        nn = keras.layers.Flatten()(nn)
        nn = keras.layers.Dense(256, activation=None, use_bias=False)(nn)
        nn = keras.layers.BatchNormalization()(nn)
        nn = keras.layers.Activation('relu')(nn)
        nn = keras.layers.Dropout(0.5)(nn)

        # output layer
        outputs = keras.layers.Dense(num_class, activation='linear',
                                     use_bias=True)(nn)

        if classification:
            outputs = keras.layers.Activation('sigmoid')(outputs)

        return keras.Model(inputs=inputs, outputs=outputs)
