# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 10:24:24 2024

@author: nikos
"""
import tensorflow as tf
from typing import Dict, Tuple, List, Text, Iterable
import numpy as np

def run_google_EPD_model(input,modelType):
    print('Attempting to load the model: EPD ', modelType)
    pathToModel=('D:\OneDrive - Imperial College London\Imperial\PhD\Google\checkpoints/'+modelType+'/epd.h5')
    model = tf.keras.models.load_model(
      pathToModel,
      compile=False,
      custom_objects={'LastChannelOneHot': LastChannelOneHot,
                      'RemoveLastChannel': RemoveLastChannel,
                      'ReshapeWithBatch': ReshapeWithBatch,
                      '_identity_activation': _identity_activation})
    # resized_data = tf.image.resize(input, (126, 126),"nearest")
    model_out = tf.cast(model(tf.convert_to_tensor(input)), tf.float64)
    return model_out.numpy()

def run_google_LSTM_model(input,modelType):
    print('Attempting to load the model: LSTM ', modelType)
    pathToModel=('D:\OneDrive - Imperial College London\Imperial\PhD\Google\checkpoints/'+modelType+'/lstm.h5')
    model = tf.keras.models.load_model(
      pathToModel,
      compile=False,
      custom_objects={'LastChannelOneHot': LastChannelOneHot,
                      'RemoveLastChannel': RemoveLastChannel,
                      'ReshapeWithBatch': ReshapeWithBatch,
                      '_identity_activation': _identity_activation})
    resized_data = tf.image.resize(input[0,:,:,:,:], (126, 126), "nearest")
    resized_data = np.expand_dims(resized_data.numpy(), axis=0)
    model_out = tf.cast(model(tf.convert_to_tensor(resized_data)), tf.float64)
    model_out_resized = tf.image.resize(model_out[0,:,:,:,:], (np.shape(input[0,0,:,:,0])))
    return np.expand_dims(model_out_resized.numpy(), axis=0)

def _identity_activation(data: tf.Tensor) -> tf.Tensor:
  """A dummy activation that is the identity."""
  return data


class RemoveLastChannel(tf.keras.layers.Layer):
  """Removes the last channel of the input tensor."""

  def call(self, network):
    return network[..., :-1]


class LastChannelOneHot(tf.keras.layers.Layer):
  """Builds a one-hot encoding for the last channel of the input tensor.

  The last channel of the input tensor is assumed to be a channel with
  categorical values that range between 0 and some max value (though possible
  encoded as floats).
  """

  def __init__(self, num_values: int = 10, **kwargs):
    super(LastChannelOneHot, self).__init__(**kwargs)
    self.num_values = num_values

  def get_config(self):
    config = super(LastChannelOneHot, self).get_config()
    config.update({'num_values': self.num_values})
    return config

  def call(self, network):
    # Strip off just the last channel and cast to an integer value.
    network = network[..., -1]
    network = tf.cast(network, dtype=tf.int32)

    #  Transform that slice into a one-hot encoding.
    network = tf.one_hot(
        network,
        off_value=0.0,
        on_value=1.0,
        depth=self.num_values,
        dtype=tf.float32)

    return network


class ReshapeWithBatch(tf.keras.layers.Layer):
  """A Reshape layer that allows reshaping to effect the batch dimension.

  The stock tf.keras.layers.Reshape layer will not allow layers to be merged
  with the batch dimension.  E.g., assume the input to the layer had a shape
  (None, 3, 4) and the desired reshape was to get to was (None, 4).  I.e., the
  2nd dimension is being merged with the batch channel (which at graph-creation
  time, is unknown in size).  There's no way to accomplish this with a Layer
  like:

    tf.layers.Reshape(target_shape=(-1, 10)) will result

  That command would not change the size at all, as it would assume the -1 size
  was fine staying at '3'.  We want the -1 to indicate that the None and 2nd
  dimension are merged into a new, larger, batch dimension.  So, instead, use
  this layer like:

    tf.layers.ReshapeWithBatch(target_shape=(-1, 10))
  """

  def __init__(self, target_shape: Iterable[int], **kwargs):
    super(ReshapeWithBatch, self).__init__(**kwargs)
    self.target_shape = target_shape

  def get_config(self):
    config = super(ReshapeWithBatch, self).get_config()
    config.update({'target_shape': self.target_shape})
    return config

  def call(self, inputs):
    return tf.reshape(inputs, shape=self.target_shape)

    
    