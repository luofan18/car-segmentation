from keras.layers import (Dense, Conv2D, Input, MaxPool2D, 
                         UpSampling2D, Concatenate, Conv2DTranspose, 
                         Dropout, Cropping2D)
from keras.initializers import he_normal
import keras.backend as K

"""
down, up, down_block, up_block are structures to construct U-net like network
for image segmentation.
"""

def down(input_layer, filters, pool=True, padding='same', verbose=1):
    conv1 = Conv2D(filters, (3, 3), padding=padding, 
                   kernel_initializer=he_normal(), activation='relu')(
                   input_layer)
    residual = Conv2D(filters, (3, 3), padding=padding, 
                      kernel_initializer=he_normal(), activation='relu')(
                      conv1)
    if pool:
        max_pool = MaxPool2D()(residual)
        if verbose:
            print ('Down output shape')
            print ('max_pool {}, residual {}'.format(K.int_shape(max_pool), K.int_shape(residual)))
        return max_pool,residual
    else:
        if verbose:
            print ('Down output shape')
            print ('residual {}'.format(K.int_shape(residual)))
        return residual
    
def up(input_layer, residual, filters, padding='same', cropping=0, verbose=1):
    filters=int(filters)
    upsample = UpSampling2D()(input_layer)
    upconv = Conv2D(filters, kernel_size=(2, 2), padding='same', 
                    kernel_initializer=he_normal(), activation='relu')(upsample)
    if cropping != 0:
        residual = Cropping2D(cropping=cropping)(residual)

    concat = Concatenate(axis=3)([residual, upconv])
    conv1 = Conv2D(filters, (3, 3), padding=padding,
                   kernel_initializer=he_normal(), activation='relu')(
                   concat)
    conv2 = Conv2D(filters, (3, 3), padding=padding, 
                   kernel_initializer=he_normal(), activation='relu')(
                   conv1)
    if verbose:
        print ('Up output dims')
        print (K.int_shape(conv2))
    return conv2

def down_block(input_layer, filters_list, dropout_list=None, padding='same'):
    residuals = []
    out = input_layer
    
    if dropout_list is None:
        dropout_list = [0] * len(filters_list)
    
    for i,filters in enumerate(filters_list):
        out, res = down(out, filters, pool=True, padding=padding)
        residuals.append(res)
        if dropout_list[i] > 0:
            out = Dropout(dropout_list[i])(out)
    return out, residuals

def up_block(input_layer, residuals, filters_list, dropout_list=None, padding='same', cropping=None):
    out = input_layer
    
    if padding == 'same':
        cropping = [0] * len(filters_list)
    if dropout_list is None:
        dropout_list = [0] * len(filters_list)
    
    for i, filters in enumerate(filters_list):
        out = up(out, residuals[::-1][i], filters, 
                 padding=padding, cropping=cropping[i])
        if dropout_list[i] > 0:
            out = Dropout(dropout[i])(out)
    return out