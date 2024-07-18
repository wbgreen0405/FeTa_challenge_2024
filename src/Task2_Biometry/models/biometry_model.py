import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv3D, MaxPooling3D, GlobalAveragePooling3D, Dense, Dropout, BatchNormalization, Activation, Input, TimeDistributed

# Define the 2D CNN base model
def build_2d_cnn_base(input_shape):
    base_model = tf.keras.applications.ResNet50(
        include_top=False, weights='imagenet', input_shape=input_shape)
    for layer in base_model.layers:
        layer.trainable = False
    return base_model

# Define the 3D CNN model using the 2D CNN base
def build_3d_cnn_with_2d_base(input_shape):
    inputs = Input(shape=input_shape)
    time_distributed = TimeDistributed(build_2d_cnn_base(input_shape[1:]))(inputs)

    x = Conv3D(64, (3, 3, 3), padding='same')(time_distributed)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling3D((2, 2, 2))(x)

    x = Conv3D(128, (3, 3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling3D((2, 2, 2))(x)

    x = GlobalAveragePooling3D()(x)

    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)

    outputs = Dense(5)(x)

    model = Model(inputs, outputs)
    return model
