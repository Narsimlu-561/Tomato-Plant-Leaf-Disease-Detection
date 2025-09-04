import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv1D

class DualECALayer(tf.keras.layers.Layer):
    def __init__(self, k_size=5, **kwargs):
        super(DualECALayer, self).__init__(**kwargs)
        self.k_size = k_size
        # Declare Conv1D layers, but don’t build them until input shape is known
        self.conv1d_sigmoid = None
        self.conv1d_tanh = None

    def build(self, input_shape):
        channels = input_shape[-1]  # number of channels in input
        # Now initialize conv1d layers after knowing input shape
        self.conv1d_sigmoid = Conv1D(
            filters=1, kernel_size=self.k_size, padding="same", use_bias=False
        )
        self.conv1d_tanh = Conv1D(
            filters=1, kernel_size=self.k_size, padding="same", use_bias=False
        )
        super(DualECALayer, self).build(input_shape)

    def call(self, x):
        # Global Average Pooling across spatial dimensions
        squeeze = tf.reduce_mean(x, axis=[1, 2])  # shape: (B, C)
        squeeze = tf.expand_dims(squeeze, axis=1)  # shape: (B, 1, C)

        # Attention path 1: Sigmoid
        attn_sigmoid = self.conv1d_sigmoid(squeeze)
        attn_sigmoid = tf.nn.sigmoid(attn_sigmoid)
        attn_sigmoid = tf.transpose(attn_sigmoid, [0, 2, 1])
        attn_sigmoid = tf.expand_dims(attn_sigmoid, axis=1)
        scale1 = x * attn_sigmoid  # Shape: (B, H, W, C)

        # Attention path 2: Tanh
        attn_tanh = self.conv1d_tanh(squeeze)
        attn_tanh = tf.nn.tanh(attn_tanh)
        attn_tanh = tf.transpose(attn_tanh, [0, 2, 1])
        attn_tanh = tf.expand_dims(attn_tanh, axis=1)
        scale2 = x * attn_tanh  # Shape: (B, H, W, C)

        # Concatenate both scaled outputs
        out = tf.concat([scale1, scale2], axis=-1)  # Concatenate along channels
        return out
