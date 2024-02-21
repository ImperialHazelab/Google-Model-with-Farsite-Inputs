from typing import Dict, Iterable, Tuple
from ml_collections import config_dict
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from helpers import Farsite2Google

FEATURE_INPUT = 'input'
FEATURE_LABEL = 'label'
# Specify the path to your TFRecordIO file
input_file = 'D:/GoogleModel/real_wn_test_temporal.tfr-00164-of-00165'
input_model = 'D:/OneDrive - Imperial College London\Imperial\PhD\Google\checkpoints/singleFuel/epd.h5'
input_row = 4


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


def get_config():
  """Provides the config for this dataset."""
  cfg = config_dict.ConfigDict()

  # The base directory where all input files are located.  This must be
  # specified for the dataset to do anything meaningful.
  cfg.input_base = 'must specify'

  # To process more than one file, this is appended to the end of input base.
  cfg.glob = '*'

  # How many data points in a single batch.
  cfg.batch_size = 1

  return cfg

def create_dataset(cfg: config_dict.ConfigDict) -> tf.data.Dataset:
  """Returns the dataset specified in the config.

  Each element in the dataset is a tuple (A, B) where A is the input data and
  B is the label data.  A and B have the same shape.  If the data is temporal,
  the output shape is (b, t, h, w, c), otherwise it's (b, h, w, c).
  b = batch, t = time step, h,w = spatial dims, c=channels.
  """
  def _data_to_dict(
      input_data: tf.Tensor,
      label_data: tf.Tensor) -> Tuple[Dict[str, tf.Tensor], tf.Tensor]:
    return ({'input_image': input_data}, label_data)

  print('DATASET PROGRESS: Inside create.  Loading file: ', cfg.input_base)
  ds = tf.data.Dataset.list_files(cfg.input_base)
  ds = ds.interleave(
      lambda x: tf.data.TFRecordDataset(x, compression_type='GZIP'),
      cycle_length=tf.data.experimental.AUTOTUNE,
      num_parallel_calls=tf.data.experimental.AUTOTUNE,
      deterministic=True)
  ds = ds.map(
      map_func=_parse_example,
      num_parallel_calls=tf.data.experimental.AUTOTUNE)
  ds = ds.map(
      map_func=_data_to_dict, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  ds = ds.batch(cfg.batch_size, drop_remainder=True)
  ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
  return ds

def _parse_example(data):
  result = tf.io.parse_single_example(
      data, get_tf_example_schema())
  feature_input = tf.io.parse_tensor(result[FEATURE_INPUT], tf.float64)
  feature_label = tf.io.parse_tensor(result[FEATURE_LABEL], tf.float64)
  return feature_input, feature_label

def get_tf_example_schema() -> Dict[str, tf.io.FixedLenFeature]:
  """Returns the example schema needed to load TFRecordDatasets."""
  return {
      FEATURE_INPUT: tf.io.FixedLenFeature([], dtype=tf.string),
      FEATURE_LABEL: tf.io.FixedLenFeature([], dtype=tf.string),
  }

cfg = get_config()
cfg.input_base = input_file
dataset = create_dataset(cfg=cfg)

rootPath = "D:\OneDrive - Imperial College London\Documents\Coding Projects\FireScenarioGenerator\FireScenarioGenerator\Output_windy/"
moistureFiles = "fms"
burn_start = [2024,1,1,1300];       "Year, Month, Day, HHMM"
burn_duration = 2;                  "Hours"
steps_per_hour = 2;                 "15-min intervals"
cellSize = 30
xllcorner = 0
yllcorner = 0

#----------Landscape file, surface-------------
#----------Create Object to save self variables-------------

fuel = Farsite2Google.get_asc_file(rootPath,'fuel.asc')
FarsiteParams = Farsite2Google(rootPath, 
                 moistureFiles, 
                 burn_duration, 
                 steps_per_hour, 
                 np.shape(fuel),
                 cellSize,
                 xllcorner,
                 yllcorner,
                 15)

"""

count = 0
for dp in dataset:
  count += 1
  if count == input_row:
    print('Found row: ', input_row)
    dp_input = dp[0]['input_image']
    dp_label = dp[1]
    print('The input shape was: ', dp_input.shape)
    print('The label shape was: ', dp_label.shape)

    print('Performing an inference...')
    pred = dp_label
    print('... done.  Prediction had the shape: ', pred.shape)

    print('The prediction shape was: ', pred.shape)
    print('The label shape was: ', dp_label.shape)

    fig, axes = plt.subplots(1, 3)
    axes[0].imshow(dp_input[0,7,:,:,14])
    axes[0].set_title('Vegetation')
    axes[1].imshow(dp_input[0,7,:,:,15])
    axes[1].set_title('Prev Front')
    axes[2].imshow(dp_input[0,7,:,:,16])
    axes[2].set_title('Scar')

    # Only perform a single inference.
    break

"""

print('Attempting to load the model from: ', input_model)
model = tf.keras.models.load_model(
      input_model,
      custom_objects={'LastChannelOneHot': LastChannelOneHot,
                      'RemoveLastChannel': RemoveLastChannel,
                      'ReshapeWithBatch': ReshapeWithBatch,
                      '_identity_activation': _identity_activation})

# Iterate through the points in the dataset.
count = 0
for dp in dataset:
  count += 1
  if count == input_row:
    print('Found row: ', input_row)
    dp_input = dp[0]['input_image']
    dp_input = dp_input[0,1,:,:,:]
    dp_input = tf.convert_to_tensor(np.expand_dims(dp_input, axis=0))
    dp_label = dp[1]
    dp_label = dp_label[0,1,:,:,:]
    dp_label = tf.convert_to_tensor(np.expand_dims(dp_label, axis=0))
    print('The input shape was: ', dp_input.shape)
    print('The label shape was: ', dp_label.shape)

    print('Performing an inference...')
    pred = tf.cast(model(dp_input), tf.float64)
    print('... done.  Prediction had the shape: ', pred.shape)

    print('The prediction shape was: ', pred.shape)
    print('The label shape was: ', dp_label.shape)

    if len(pred.shape) == 5:
      # Temporal data, the final prediction is the last time step.
      pred = pred[0,-1,:,:,0]
      dp_label = dp_label[0,-1,:,:,0]
    elif len(pred.shape) == 4:
      # Static data, the final prediction is the final prediction.
      pred = pred[0,:,:,0]
      dp_label = dp_label[0,:,:,0]
    else:
      raise ValueError('The prediction has the wrong shape.')

    fig, axes = plt.subplots(1, 3)
    axes[0].imshow(dp_label)
    axes[0].set_title('Label')
    axes[1].imshow(dp_input[0,:,:,2])
    axes[1].set_title('Prediction')
    error = dp_label - pred
    axes[2].imshow(error)
    axes[2].set_title('Error')

    # Only perform a single inference.
    break

denormedOutput = np.zeros(np.shape(dp_input[0,7,:,:,:]))

for i in range(0,17):
    denormedOutput[:,:,i] = Farsite2Google.denorm_data_by_norms(dp_input[0,7,:,:,i],"singleFuel",i)

FarsiteParams.channels2excel(dp_input[0,7,:,:,:],"channels_Google.xlsx")
FarsiteParams.channels2excel(denormedOutput[:,:,:],"channels_Google_Denormalised.xlsx")


