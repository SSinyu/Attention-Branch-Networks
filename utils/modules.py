from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras import activations, Model, Input


def basic_block(x, out_ch, kernel_size=3, stride=1, last_act=True):
    """
    (batch, height, width, channels) => (batch, heigth, width, out_ch)
    """
    bn_axis = 3 if K.image_data_format() == 'channels_last' else 1
    ep = 1.001e-5

    out = layers.Conv2D(out_ch, kernel_size, stride, 'same', use_bias=False)(x)
    out = layers.BatchNormalization(axis=bn_axis, epsilon=ep)(out)

    if last_act is True:
        return layers.Activation(activations.relu)(out)
    else:
        return out


def bottleneck_block(x, out_ch, stride=1):
    """
    (batch, height, width, channels) =>
    stride == 1, (batch, height, width, out_ch)
    stride == 2, (batch, height/2, width/2, out_ch)
    """
    # if x._shape_tuple()[-1] != out_ch:
    if int(x.shape[-1]) != out_ch or stride == 2:
        bn_axis = 3 if K.image_data_format() == 'channels_last' else 1
        ep = 1.001e-5
        shortcut = layers.Conv2D(out_ch, 1, stride, 'same', use_bias=False)(x)
        shortcut = layers.BatchNormalization(axis=bn_axis, epsilon=ep)(shortcut)
    else:
        shortcut = x

    out = basic_block(x, out_ch//4, 1)
    out = basic_block(out, out_ch//4, 3, stride)
    out = basic_block(out, out_ch, 1, last_act=False)
    out = layers.Add()([out, shortcut])
    return layers.Activation(activations.relu)(out)


def feature_extractor(x, out_ch, n):
    """
    (batch, height, width, channels) =>
        (batch, height/2, width/2, out_ch)
    """
    out = basic_block(x, out_ch)
    out = bottleneck_block(out, out_ch, 1)
    for _ in range(n-1):
        out = bottleneck_block(out, out_ch)

    out = bottleneck_block(out, out_ch*2, 2)
    for _ in range(n-1):
        out = bottleneck_block(out, out_ch*2)
    return out


def attention_branch(x, n, n_classes, name='attention_branch'):
    """
    (batch, height, width, channels) =>
        (batch, n_classes), (batch, height, width, channels)
    """
    bn_axis = 3 if K.image_data_format() == 'channels_last' else 1
    ep = 1.001e-5

    filters = x._shape_tuple()[-1]
    out = bottleneck_block(x, filters*2, 1) # (b,h/2,w/2,f*2*4*2)
    for _ in range(n-1):
        out = bottleneck_block(out, filters*2, 1) # (b,h/2,w/2,f*2*4*2)

    out = layers.BatchNormalization(axis=bn_axis, epsilon=ep, name=name+'_bn_1')(out)
    out = layers.Conv2D(n_classes, 1, 1, 'same', use_bias=False, activation=activations.relu ,name=name+'_conv_1')(out)

    pred_out = layers.Conv2D(n_classes, 1, 1, 'same', use_bias=False, name=name+'_pred_conv_1')(out)
    pred_out = layers.GlobalAveragePooling2D(name=name+'_gap_1')(pred_out)
    pred_out = layers.Softmax(name='attention_branch_output')(pred_out)

    att_out = layers.Conv2D(1, 1, 1, 'same', use_bias=False, name=name+'_att_conv_1')(out)
    att_out = layers.BatchNormalization(axis=bn_axis, epsilon=ep, name=name+'_att_bn_1')(att_out)
    att_out = layers.Activation(activations.sigmoid, name=name+'_att_sigmoid_1')(att_out)
    # att_out = (x * att_out) + x
    att_out = layers.Lambda(lambda z: (z[0] * z[1]) + z[0])([x, att_out])
    return pred_out, att_out


def perception_branch(x, n, n_classes, name='perception_branch'):
    filters = x._shape_tuple()[-1]
    out = bottleneck_block(x, filters*2, 1)
    for _ in range(n-1):
        out = bottleneck_block(out, filters*2, 1)

    out = layers.GlobalAveragePooling2D(name=name+'_avgpool_1')(out)
    out = layers.Dense(256, name=name+'_dense_1')(out)
    out = layers.Dense(n_classes, name=name+'_dense_2')(out)
    return layers.Softmax(name='perception_branch_output')(out)


def get_model(input_shape, n_classes, out_ch=256, n=18):
    img_input = Input(shape=input_shape, name='input_image')

    backbone = feature_extractor(img_input, out_ch, n)
    att_pred, att_map = attention_branch(backbone, n, n_classes)
    per_pred = perception_branch(att_map, n, n_classes)

    model = Model(inputs=img_input, outputs=[att_pred, per_pred])
    return model
