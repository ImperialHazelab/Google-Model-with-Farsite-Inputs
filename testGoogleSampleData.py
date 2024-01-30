import os
from typing import Dict, Iterable, Tuple
from ml_collections import config_dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from helpers import Farsite2Google

FEATURE_INPUT = 'input'
FEATURE_LABEL = 'label'
# Specify the path to your TFRecordIO file
input_file = 'D:/GoogleModel/wildfire_conv_ltsm/CLI_Stuff/real_wn_test_temporal.tfr-00164-of-00165'
input_row = 4


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
                 burn_start, 
                 burn_duration, 
                 steps_per_hour, 
                 np.shape(fuel),
                 cellSize,
                 xllcorner,
                 yllcorner)

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

FarsiteParams.channels2excel(dp_input[0,7,:,:,:],"channels_Google.xlsx")



