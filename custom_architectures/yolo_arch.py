import os
import numpy as np
import tensorflow as tf

from custom_architectures.current_blocks import Conv2DBN

FULL_YOLO_BACKEND_PATH  = "pretrained_models/yolo_backend/full_yolo_backend.h5"
TINY_YOLO_BACKEND_PATH  = "pretrained_models/yolo_backend/tiny_yolo_backend.h5"

def FullYoloBackend(input_image,
                    weight_path = FULL_YOLO_BACKEND_PATH,
                   
                    name = 'feature_extractor'
                   ):
    # the function to implement the orgnization layer (thanks to github.com/allanzelener/YAD2K)
    def space_to_depth_x2(x):
        import tensorflow as tf
        return tf.nn.space_to_depth(x, block_size = 2)
    
    if isinstance(input_image, int): input_image = (input_image, input_image, 3)
    if isinstance(input_image, tuple):
        input_image = tf.keras.layers.Input(shape = input_image, name = 'input_image')


    x = input_image
    # Layers 1 and 2
    for i, filters in enumerate([32, 64]):
        x = Conv2DBN(
            x, filters = filters, kernel_size = (3,3), padding = 'same', use_bias = False,

            bnorm       = 'after', pooling = 'max', drop_rate = 0.,
            activation  = 'leaky', activation_kwargs   = {'alpha' : 0.1},

            name = 'conv_{}'.format(i+1)
        )
        
    # Layer 3
    x = Conv2DBN(
        x, filters = 128, kernel_size = (3,3), padding = 'same', use_bias = False,

        bnorm       = 'after', pooling = None, drop_rate = 0.,
        activation  = 'leaky', activation_kwargs   = {'alpha' : 0.1},

        name = 'conv_3'
    )


    # Layer 4
    x = Conv2DBN(
        x, filters = 64, kernel_size = (1,1), padding = 'same', use_bias = False,

        bnorm       = 'after', pooling = None, drop_rate = 0.,
        activation  = 'leaky', activation_kwargs   = {'alpha' : 0.1},

        name = 'conv_4'
    )

    # Layer 5
    x = Conv2DBN(
        x, filters = 128, kernel_size = (3,3), padding = 'same', use_bias = False,

        bnorm       = 'after', pooling = 'max', drop_rate = 0.,
        activation  = 'leaky', activation_kwargs   = {'alpha' : 0.1},

        name = 'conv_5'
    )

    # Layer 6
    x = Conv2DBN(
        x, filters = 256, kernel_size = (3,3), padding = 'same', use_bias = False,

        bnorm       = 'after', pooling = None, drop_rate = 0.,
        activation  = 'leaky', activation_kwargs   = {'alpha' : 0.1},

        name = 'conv_6'
    )

    # Layer 7
    x = Conv2DBN(
        x, filters = 128, kernel_size = (1,1), padding = 'same', use_bias = False,

        bnorm       = 'after', pooling = None, drop_rate = 0.,
        activation  = 'leaky', activation_kwargs   = {'alpha' : 0.1},

        name = 'conv_7'
    )

    # Layer 8
    x = Conv2DBN(
        x, filters = 256, kernel_size = (3,3), padding = 'same', use_bias = False,

        bnorm       = 'after', pooling = 'max', drop_rate = 0.,
        activation  = 'leaky', activation_kwargs   = {'alpha' : 0.1},

        name = 'conv_8'
    )

    # Layers 9 to 13
    for i, (filters, kernel) in enumerate([(512, (3,3)), (256, (1,1)), (512, (3,3)), (256, (1,1)), (512, (3,3))]):
        x = Conv2DBN(
            x, filters = filters, kernel_size = kernel,
            padding = 'same', use_bias = False,

            bnorm       = 'after', pooling = None, drop_rate = 0.,
            activation  = 'leaky', activation_kwargs   = {'alpha' : 0.1},

            name = 'conv_{}'.format(9 + i)
        )


    skip_connection = x

    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

    for i, (filters, kernel) in enumerate([(1024, (3,3)), (512, (1,1)), (1024, (3,3)), (512, (1,1)), (1024, (3,3)), (1024, (3,3)), (1024, (3,3))]):
        x = Conv2DBN(
            x, filters = filters, kernel_size = kernel,
            padding = 'same', use_bias = False,

            bnorm       = 'after', pooling = None, drop_rate = 0.,
            activation  = 'leaky', activation_kwargs   = {'alpha' : 0.1},

            name = 'conv_{}'.format(14 + i)
        )

    # Layer 21
    skip_connection = Conv2DBN(
        skip_connection, filters = 64, kernel_size = (1,1),
        padding = 'same', use_bias = False,

        bnorm = 'after', pooling = None, drop_rate = 0.,
        activation  = 'leaky', activation_kwargs   = {'alpha' : 0.1},

        name = 'conv_21'
    )
    skip_connection = tf.keras.layers.Lambda(space_to_depth_x2)(skip_connection)

    x = tf.keras.layers.Concatenate()([skip_connection, x])

    # Layer 22
    output = Conv2DBN(
        x, filters = 1024, kernel_size = (3,3), padding = 'same', use_bias = False,

        bnorm       = 'after', pooling = None, drop_rate = 0.,
        activation  = 'leaky', activation_kwargs   = {'alpha' : 0.1},

        name = 'conv_22'
    )
    
    model = tf.keras.Model(inputs = input_image, outputs = output, name = name)

    if weight_path is not None: #and os.path.exists(weight_path):
        print("Loading weights from {}".format(weight_path))
        model.load_weights(weight_path)

    return model

def TinyYoloBackend(input_image,
                    weight_path = FULL_YOLO_BACKEND_PATH,
                   
                    name = 'feature_extractor'
                   ):
    if isinstance(input_image, int): input_image = (input_image, input_image, 3)
    if isinstance(input_image, tuple):
        input_image = tf.keras.layers.Input(shape = input_image, name = 'input_image')

    # Layer 1
    x = Conv2DBN(
        input_image, filters = 16, kernel_size = (3,3),
        padding = 'same', use_bias = False,

        bnorm       = 'after', pooling = 'max', drop_rate = 0.,
        activation  = 'leaky', activation_kwargs   = {'alpha' : 0.1},

        name = 'conv_1'
    )

    # Layer 2 - 5
    for i in range(4):
        x = Conv2DBN(
            x, filters = 32 * (2**i), kernel_size = (3,3),
            padding = 'same', use_bias = False,

            bnorm       = 'after', pooling = 'max', drop_rate = 0.,
            activation  = 'leaky', activation_kwargs   = {'alpha' : 0.1},

            name = 'conv_{}'.format(i+2)
        )

    # Layer 6
    x = Conv2DBN(
        x, filters = 512, kernel_size = (3,3), padding = 'same', use_bias = False,

        bnorm       = 'after', drop_rate = 0.,
        pooling = 'max', pool_strides = (1,1), pool_padding = 'same',
        activation  = 'leaky', activation_kwargs   = {'alpha' : 0.1},

        name = 'conv_6'
    )

    # Layer 7 - 8
    for i in range(0,2):
        x = Conv2DBN(
            x, filters = 1024, kernel_size = (3,3), padding = 'same', use_bias = False,

            bnorm       = 'after', pooling = None, drop_rate = 0.,
            activation  = 'leaky', activation_kwargs   = {'alpha' : 0.1},

            name = 'conv_{}'.format(i+7)
        )

    model = tf.keras.Model(inputs = input_image, outputs = x, name = name)

    if weight_path is not None and os.path.exists(weight_path):
        model.load_weights(weight_path)

    return model

def YOLO(feature_extractor,
         nb_class,
         nb_box     = 5,
         input_size = None,
         flatten    = True,
         randomize  = True,
         name       = 'yolo',
         ** kwargs
        ):
    assert isinstance(feature_extractor, tf.keras.Model)
    
    if isinstance(input_size, int): input_size = (input_size, input_size, 3)
    
    if input_size is None or input_size == feature_extractor.input.shape[1:] and flatten:
        input_image = feature_extractor.input
        features    = feature_extractor.output
    else:
        input_image = tf.keras.layers.Input(shape = input_size, name = 'input_image')
        features    = feature_extractor(input_image)
    
    grid_h, grid_w = features.shape[1:3]     
    
    # make the object detection layer
    output = tf.keras.layers.Conv2D(
        filters = nb_box * (4 + 1 + nb_class), kernel_size = (1,1), padding = 'same', 
        kernel_initializer = 'lecun_normal', name = 'detection_layer'
    )(features)
    
    output = tf.keras.layers.Reshape((grid_h, grid_w, nb_box, 4 + 1 + nb_class))(output)

    model = tf.keras.Model(inputs = input_image, outputs = output, name = name)

    if randomize:
        # initialize the weights of the detection layer
        layer = model.layers[-2]
        kernel, bias = layer.get_weights()

        new_kernel = np.random.normal(size = kernel.shape)  / (grid_h * grid_w)
        new_bias   = np.random.normal(size = bias.shape)    / (grid_h * grid_w)

        layer.set_weights([new_kernel, new_bias])

    return model

custom_functions    = {
    'full_yolo' : FullYoloBackend,
    'FullYolo'  : FullYoloBackend,
    'tiny_yolo' : TinyYoloBackend,
    'TinyYolo'  : TinyYoloBackend,
    'YOLO'  : YOLO
}
