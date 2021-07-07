#Resnet50 implementation

def create_identity_block(x, f1, f2, f3, filter_size):
    temp = x

    #first lyaer
    x = tf.keras.layers.Conv2D(filters = f1, kernel_size = (1, 1), strides = (1,1), padding = 'same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    
    #second layer
    x = tf.keras.layers.Conv2D(filters = f2, kernel_size = filter_size, strides = (1,1), padding = 'same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    
    #third layer
    x = tf.keras.layers.Conv2D(filters = f3, kernel_size = (1, 1), strides = (1,1), padding = 'same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    #skip connection
    x = tf.keras.layers.Add([x, temp])
    x = tf.keras.layers.Activation('relu')(x)
    return x
  
def create_convolutional_block(x, f1, f2, f3, filter_size):
    #input size != output size
    temp = x
    
    #first layer
    x = tf.keras.layers.Conv2D(filters = f1, kernel_size = (1, 1), strides = (1,1), padding = 'same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    
    #second layer
    x = tf.keras.layers.Conv2D(filters = f2, kernel_size = filter_size, strides = (1,1), padding = 'same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    
    x = tf.keras.layers.Conv2D(filters = f3, kernel_size = (1, 1), strides = (1,1), padding = 'same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    #add convolutional block
    temp = tf.keras.layers.Conv2D(filters = f3, kernel_size = (1,1), strides = (2,2), padding = 'same')(temp)
    temp = tf.keras.layers.BatchNormalization()(temp)
    
    #skip connection
    x = tf.keras.layers.Add([x, temp])
    x = tf.keras.layers.Activation('relu')(x)
    
def Resnet(data):
    #input_size = (224,224,3)
    #zero padding - to preserve information 
    #7*7, 64
    
    x = create_convolutional_block(x, 64, 64, 256, (3, 3))
    x = create_identity_block(x, 64, 64, 256, (3, 3))
    x = create_identity_block(x, 64, 64, 256, (3, 3))
    
    x = create_convolutional_block(x, 128, 128, 512, (3, 3))
    x = create_identity_block(x, 64, 64, 256, (3, 3))
    x = create_identity_block(x, 64, 64, 256, (3, 3))
    x = create_identity_block(x, 64, 64, 256, (3, 3))
    
    x = create_convolutional_block(x, 256, 256, 1024, (3, 3))
    x = create_identity_block(x, 256, 256, 1024, (3, 3))
    x = create_identity_block(x, 256, 256, 1024, (3, 3))
    x = create_identity_block(x, 256, 256, 1024, (3, 3))
    x = create_identity_block(x, 256, 256, 1024, (3, 3))
    
    x = create_convolutional_block(x, 512, 512, 2048, (3, 3))
    x = create_identity_block(x, 512, 512, 2048, (3, 3))
    x = create_identity_block(x, 512, 512, 2048, (3, 3))
    
    x = tf.keras.layers.  AveragePooling2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.flatten()(x)

    return tf.keras.Model(input = data, outputs = x, name = 'ResNet50')
