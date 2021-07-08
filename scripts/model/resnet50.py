import tensorflow as tf
import tensorflow.keras.layers
from keras.datasets import cifar10

#ResNet50 implementation

class ResNet50:
    
    def create_identity_block(x, f1, f2, f3, filter_size):
        temp = x

        #first layer
        x = tf.keras.layers.Conv2D(filters = f1, kernel_size = (1, 1), strides = (1,1), padding = 'same')(x)
        x = tf.keras.layers.BatchNormalization(axis = 3)(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Activation('relu')(x)

        #second layer
        x = tf.keras.layers.Conv2D(filters = f2, kernel_size = filter_size, strides = (1,1), padding = 'same')(x)
        x = tf.keras.layers.BatchNormalization(axis = 3)(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Activation('relu')(x)

        #third layer
        x = tf.keras.layers.Conv2D(filters = f3, kernel_size = (1, 1), strides = (1,1), padding = 'same')(x)
        x = tf.keras.layers.BatchNormalization(axis = 3)(x)
        x = tf.keras.layers.Dropout(0.5)(x)

        #skip connection
        x = tf.keras.layers.Add()([x, temp])
        x = tf.keras.layers.Activation('relu')(x)

        return x

    def create_convolutional_block(x, f1, f2, f3, filter_size, strides):
        #input size != output size
        #resize the input by using 1*1 convolution

        temp = x

        #first layer
        x = tf.keras.layers.Conv2D(filters = f1, kernel_size = (1, 1), strides = (strides, strides), padding = 'same')(x)
        x = tf.keras.layers.BatchNormalization(axis = 3)(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Activation('relu')(x)

        #second layer
        x = tf.keras.layers.Conv2D(filters = f2, kernel_size = filter_size, strides = (1,1), padding = 'same')(x)
        x = tf.keras.layers.BatchNormalization(axis = 3)(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Activation('relu')(x)

        x = tf.keras.layers.Conv2D(filters = f3, kernel_size = (1, 1), strides = (1,1), padding = 'same')(x)
        x = tf.keras.layers.BatchNormalization()(x)

        #add convolutional block
        temp = tf.keras.layers.Conv2D(filters = f3, kernel_size = (1,1), strides = (strides, strides), padding = 'same')(temp)
        x = tf.keras.layers.BatchNormalization(axis = 3)(x)
        x = tf.keras.layers.Dropout(0.5)(x)

        #skip connection
        x = tf.keras.layers.Add()([x, temp])
        x = tf.keras.layers.Activation('relu')(x)

        return x

    def Resnet(input_shape, num_classes):
        #zero padding - to preserve information     
        #instantiate a Keras tensor
        input_t = tf.keras.Input(input_shape) #shape=(None, 32, 32, 3)
        x = tf.keras.layers.ZeroPadding2D((3, 3))(input_t) #shape=(None, 38, 38, 3)

        #7*7, 64
        x = tf.keras.layers.Conv2D(64, kernel_size = (7, 7), strides = (2, 2), padding = 'same')(x)
        x = tf.keras.layers.BatchNormalization(axis = 3)(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.MaxPooling2D((3, 3), strides = (2, 2))(x)

        x = create_convolutional_block(x, 64, 64, 256, (3, 3), strides = 1)
        x = create_identity_block(x, 64, 64, 256, (3, 3))
        x = create_identity_block(x, 64, 64, 256, (3, 3))

        x = create_convolutional_block(x, 128, 128, 512, (3, 3), strides = 2)
        x = create_identity_block(x, 128, 128, 512, (3, 3))
        x = create_identity_block(x, 128, 128, 512, (3, 3))
        x = create_identity_block(x, 128, 128, 512, (3, 3))

        x = create_convolutional_block(x, 256, 256, 1024, (3, 3), strides = 2)
        x = create_identity_block(x, 256, 256, 1024, (3, 3))
        x = create_identity_block(x, 256, 256, 1024, (3, 3))
        x = create_identity_block(x, 256, 256, 1024, (3, 3))
        x = create_identity_block(x, 256, 256, 1024, (3, 3))
        x = create_identity_block(x, 256, 256, 1024, (3, 3))

        x = create_convolutional_block(x, 512, 512, 2048, (3, 3), strides = 2)
        x = create_identity_block(x, 512, 512, 2048, (3, 3))
        x = create_identity_block(x, 512, 512, 2048, (3, 3))

        x = tf.keras.layers.AveragePooling2D(pool_size = 2)(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Flatten()(x)

        #dropout? 
        x = tf.keras.layers.Dense(num_classes, activation = 'softmax')(x) #kernel_initializer

        return tf.keras.Model(inputs = input_t, outputs = x, name = 'ResNet50')

    
        