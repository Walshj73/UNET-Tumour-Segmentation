# Import the required libraries.
from tensorflow.keras.layers import (
        Activation,
        Conv2D,
        Conv2DTranspose,
        Input,
        BatchNormalization,
        Add,
        MaxPool2D
)
from tensorflow.keras.models import Model


# n
encoder_output_maps = {
    'block_1': 64,
    'block_2': 128,
    'block_3': 256,
    'block_4': 512
}
decoder_output_maps = {
    'block_1': 64,
    'block_2': 64,
    'block_3': 128,
    'block_4': 256
}

# m
encoder_input_maps = {
    'block_1': 64,
    'block_2': 64,
    'block_3': 128,
    'block_4': 256
}
decoder_input_maps = {
    'block_1': 64,
    'block_2': 128,
    'block_3': 256,
    'block_4': 512
}


# Function to create the LinkNet architecture.
def LinkNet(img_height, img_width, nclasses=None):

    def encoder_block(input_tensor, block=None):
        block_name = f'block_{block}'
        nfilters = encoder_output_maps[block_name]

        input_tensor_projection = Conv2D(
                filters=nfilters,
                kernel_size=(1, 1),
                strides=(2, 2),
                padding='same',
                kernel_initializer='he_normal',
                name=f'{block_name}_conv2d_1x1'
        )(input_tensor)
        input_tensor_projection = BatchNormalization(
                name=f'{block_name}_bn_1x1'
        )(input_tensor_projection)
        input_tensor_projection = Activation(
                'relu',
                name=f'{block_name}_relu_1x1'
        )(input_tensor_projection)

        x = Conv2D(
                filters=nfilters,
                kernel_size=(3, 3),
                strides=(2, 2),
                padding='same',
                kernel_initializer='he_normal',
                name=f'encoder_{block_name}_conv2d_1'
        )(input_tensor)
        x = BatchNormalization(name=f'encoder_{block_name}_bn_1')(x)
        x = Activation('relu', name=f'encoder_{block_name}_relu_1')(x)
        x = Conv2D(
                filters=nfilters,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding='same',
                kernel_initializer='he_normal',
                name=f'encoder_{block_name}_conv2d_2'
        )(x)
        x = BatchNormalization(name=f'encoder_{block_name}_bn_2')(x)
        x = Add(
            name=f'encoder_{block_name}_add_1'
        )([x, input_tensor_projection])
        x = Activation('relu', name=f'encoder_{block_name}_relu_2')(x)
        x_res = x

        x = Conv2D(
                filters=nfilters,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding='same',
                kernel_initializer='he_normal',
                name=f'encoder_{block_name}_conv2d_3'
        )(x_res)
        x = BatchNormalization(name=f'encoder_{block_name}_bn_3')(x)
        x = Activation('relu', name=f'encoder_{block_name}_relu_3')(x)
        x = Conv2D(
                filters=nfilters,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding='same',
                kernel_initializer='he_normal',
                name=f'encoder_{block_name}_conv2d_4'
        )(x)
        x = BatchNormalization(name=f'encoder_{block_name}_bn_4')(x)
        x = Add(name=f'encoder_{block_name}_add_2')([x, x_res])
        x = Activation('relu', name=f'encoder_{block_name}_relu_4')(x)
        return x

    def decoder_block(input_tensor, block=None):
        block_name = f'block_{block}'
        nfilters_b = decoder_output_maps[block_name]
        nfilters_a = decoder_input_maps[block_name] // 4

        x = Conv2D(
                filters=nfilters_a,
                kernel_size=(1, 1),
                strides=(1, 1),
                padding='same',
                kernel_initializer='he_normal',
                name=f'decoder_{block_name}_conv2d_1x1'
        )(input_tensor)
        x = BatchNormalization(name=f'decoder_{block_name}_bn_1')(x)
        x = Activation('relu', name=f'decoder_{block_name}_relu_1')(x)
        x = Conv2DTranspose(
                filters=nfilters_a,
                kernel_size=(3, 3),
                strides=(2, 2),
                padding='same',
                kernel_initializer='he_normal',
                name=f'decoder_{block_name}_conv2dT_1'
        )(x)
        x = BatchNormalization(name=f'decoder_{block_name}_bn_2')(x)
        x = Activation('relu', name=f'decoder_{block_name}_relu_2')(x)
        x = Conv2D(
                filters=nfilters_b,
                kernel_size=(1, 1),
                strides=(1, 1),
                padding='same',
                kernel_initializer='he_normal',
                name=f'decoder_{block_name}_conv2d_1x2'
        )(x)
        x = BatchNormalization(name=f'decoder_{block_name}_bn_3')(x)
        x = Activation('relu', name=f'decoder_{block_name}_relu_3')(x)
        return x

    input_layer = Input(shape=(img_height, img_width, 3), name='image_input')
    y = Conv2D(
            filters=64,
            kernel_size=(7, 7),
            strides=(2, 2),
            padding='same',
            kernel_initializer='he_normal',
            name='initial_conv2d'
    )(input_layer)
    y = BatchNormalization(name='initial_block_bn')(y)
    y = Activation('relu', name='initial_block_relu')(y)
    y = MaxPool2D(pool_size=(2, 2), padding='same',
                  name='initial_block_maxpool')(y)
    encoder_outputs = [None]
    for i in range(1, 5):
        y = encoder_block(input_tensor=y, block=i)
        encoder_outputs.append(y)
    for i in range(4, 1, -1):
        y = decoder_block(input_tensor=y, block=i)
        y = Add()([y, encoder_outputs[i - 1]])
    y = decoder_block(input_tensor=y, block=1)

    y = Conv2DTranspose(
            filters=32,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding='same',
            kernel_initializer='he_normal',
            name='final_conv2dT'
    )(y)
    y = BatchNormalization(name='final_block_bn_1')(y)
    y = Activation('relu', name='final_block_relu_1')(y)
    y = Conv2D(
            filters=32,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            kernel_initializer='he_normal',
            name='final_conv2d'
    )(y)
    y = BatchNormalization(name='final_block_bn_2')(y)
    y = Activation('relu', name='final_block_relu_2')(y)
    y = Conv2DTranspose(
            filters=nclasses,
            kernel_size=(2, 2),
            strides=(2, 2),
            padding='same',
            kernel_initializer='he_normal',
            name='output_conv2dT'
    )(y)
    output_layer = Activation('sigmoid', name='predictions')(y)

    model = Model(inputs=input_layer, outputs=output_layer, name='LinkNet')
    return model


if __name__ == "__main__":
    model = LinkNet(128, 128, 1)
    model.summary()
