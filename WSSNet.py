import tensorflow as tf
from keras import layers, ops as Kops

class WSSNet():
    def build_network(self, input_layer):
        print('UNet BN')
        padding = 'SAME'
        channel_nr = 64
        
        [xyz0, xyz1, xyz2, v1, v2] = input_layer

        input_layer = tf.keras.layers.concatenate([xyz0, xyz1, xyz2, v1, v2])
        
        # === Starting U-Net ===
        # =========== Downward blocks =========== 
        conv1 = conv2d(input_layer, kernel_size=3, filters=channel_nr, padding='PERIODIC', activation='relu')
        conv1 = conv2d(conv1, kernel_size=3, filters=channel_nr, padding='PERIODIC', activation='relu')
        conv1 = tf.keras.layers.BatchNormalization()(conv1)
        pool1 = tf.keras.layers.MaxPooling2D()(conv1)

        conv2 = conv2d(pool1, kernel_size=3, filters=channel_nr * 2, padding=padding, activation='relu')
        conv2 = conv2d(conv2, kernel_size=3, filters=channel_nr * 2, padding=padding, activation='relu')
        conv2 = tf.keras.layers.BatchNormalization()(conv2)
        pool2 = tf.keras.layers.MaxPooling2D()(conv2)

        conv3 = conv2d(pool2, kernel_size=3, filters=channel_nr * 4, padding=padding, activation='relu')
        conv3 = conv2d(conv3, kernel_size=3, filters=channel_nr * 4, padding=padding, activation='relu')
        conv3 = tf.keras.layers.BatchNormalization()(conv3)
        pool3 = tf.keras.layers.MaxPooling2D()(conv3)

        # =========== bottom layer =========== 
        conv4 = conv2d(pool3, kernel_size=3, filters=channel_nr * 8, padding=padding, activation='relu')
        conv4 = conv2d(conv4, kernel_size=3, filters=channel_nr * 8, padding=padding, activation='relu')
        conv4 = tf.keras.layers.BatchNormalization()(conv4)

        # =========== Upward blocks =========== 
        up5 = tf.keras.layers.UpSampling2D(size=(2,2), interpolation='bilinear')(conv4)
        up5 = conv2d(up5, kernel_size=3, filters=channel_nr * 4, padding=padding, activation='relu')
        merge5 = tf.keras.layers.concatenate([conv3,up5])

        conv5 = conv2d(merge5, kernel_size=3, filters=channel_nr * 4, padding=padding, activation='relu')
        conv5 = conv2d(conv5, kernel_size=3, filters=channel_nr * 4, padding=padding, activation='relu')
        conv5 = tf.keras.layers.BatchNormalization()(conv5)


        up6 = tf.keras.layers.UpSampling2D(size=(2,2), interpolation='bilinear')(conv5)
        up6 = conv2d(up6, kernel_size=3, filters=channel_nr * 2, padding=padding, activation='relu')
        merge6 = tf.keras.layers.concatenate([conv2,up6])

        conv6 = conv2d(merge6, kernel_size=3, filters=channel_nr * 2, padding=padding, activation='relu')
        conv6 = conv2d(conv6, kernel_size=3, filters=channel_nr * 2, padding=padding, activation='relu')
        conv6 = tf.keras.layers.BatchNormalization()(conv6)

        up7 = tf.keras.layers.UpSampling2D(size=(2,2), interpolation='bilinear')(conv6)
        up7 = conv2d(up7, kernel_size=3, filters=channel_nr, padding=padding, activation='relu')
        merge7 = tf.keras.layers.concatenate([conv1,up7])

        conv7 = conv2d(merge7, kernel_size=3, filters=channel_nr, padding=padding, activation='relu')
        conv7 = conv2d(conv7, kernel_size=3, filters=channel_nr, padding=padding, activation='relu')
        
        # output layer, 3 channels (vector)
        wss = conv2d(conv7, kernel_size=3, filters=3, padding=padding, activation=None)
    
        
        return wss
    
def conv2d(x, kernel_size=3, filters=16, padding='SAME', activation='relu'):
    p = (kernel_size - 1) // 2
    if padding.upper() == 'PERIODIC':
        # periodic padding on H and W
        x = periodic_padding_flexible(x, axis=1, padding=p)
        x = periodic_padding_flexible(x, axis=2, padding=p)
        pad_mode = 'valid'
    elif padding.upper() == 'SYMMETRIC':
        x = symmetric_padding(x, pad_h=p, pad_w=p)
        pad_mode = 'valid'
    else:
        pad_mode = 'same'
    x = layers.Conv2D(filters, kernel_size, padding=pad_mode, activation=activation)(x)
    return x
def periodic_padding_flexible(tensor, axis=1, padding=1):
    """Keras-safe periodic padding using Lambda (works with KerasTensor)."""
    def _pad_fn(x):
        import tensorflow as tf
        if padding == 0:
            return x
        if axis == 1:  # pad height (H)
            left  = x[:, -padding:, :, :]
            mid   = x
            right = x[:, :padding, :, :]
            return tf.concat([left, mid, right], axis=1)
        elif axis == 2:  # pad width (W)
            left  = x[:, :, -padding:, :]
            mid   = x
            right = x[:, :, :padding, :]
            return tf.concat([left, mid, right], axis=2)
        else:
            raise ValueError("axis must be 1 (H) or 2 (W)")
    return layers.Lambda(_pad_fn)(tensor)

def symmetric_padding(tensor, pad_h=1, pad_w=1):
    """SYMMETRIC padding via tf.pad wrapped in a Keras Lambda."""
    def _sym(x):
        import tensorflow as tf
        pads = [[0,0],[pad_h,pad_h],[pad_w,pad_w],[0,0]]
        return tf.pad(x, pads, mode='SYMMETRIC')
    return layers.Lambda(_sym)(tensor)
