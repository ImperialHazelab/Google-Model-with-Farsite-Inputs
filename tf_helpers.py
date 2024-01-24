# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 21:38:53 2024

@author: nikos
"""
import tensorflow as tf
from typing import Dict, Tuple, List, Text
import numpy as np

class tfHelpers:
    def convert_to_tensor():
        for instance in tfHelpers.generate_time_instances(
            results, self.cfg.tree_noise_mean, self.cfg.tree_noise_std,
            skip_initial_rows=self.cfg.skip_initial_rows):
          self._data_static.inc()
          yield tfHelpers.get_tf_example(
              instance, set_label, movie_idx)
        
        # Create the temporal data points.
        for sequence in tfHelpers.generate_time_sequences(
            results, self.cfg.sequence_length, self.cfg.tree_noise_mean,
            self.cfg.tree_noise_std, self.cfg.skip_initial_rows):
          self._data_temporal.inc()
          yield tfHelpers.get_tf_example(sequence, set_label, movie_idx)
    
    
    
    def generate_rows(data: List[any], noise_mean: float, noise_std: float):
      """Yields individual rows out of arrays in data[0] and data[1].
    
      Args:
        data: The data.
        noise_mean: Mean of normal noise generator.
        noise_std: Standard deviation of normal noise generator.
    
      Yields:
        The input and label frame with shapes (1, h, w, c_input) and (1, h, w,
        c_label).
      """
      for i in range(data[0].shape[0]):
        input_frame = data[0][i:i+1, ::, ::, ::]  # (1, h, w, c)
        if noise_std != 0.0:
          height = input_frame.shape[1]
          width = input_frame.shape[2]
          num_channels = input_frame.shape[3]
          input_frame_split = tf.split(input_frame, num_channels, axis=3)
          noise_frame = tf.random.normal(
              shape=(1, height, width, 1),
              mean=noise_mean,
              stddev=noise_std,
              dtype=tf.dtypes.float64)
          noise_frame = tf.clip_by_value(noise_frame, 0.0, 100000.0)
          input_frame_split[0] = tf.math.multiply(
              input_frame_split[0], noise_frame)
          input_frame = tf.concat(input_frame_split, 3)
        label_frame = data[1][i:i+1, ::, ::, ::]
        yield (input_frame, label_frame)
    
    
    def generate_time_sequences(data,
                                time_length: int,
                                noise_mean: float,
                                noise_std: float,
                                skip_initial_rows: int = -1):
      """Generates input/label pairs with a time dimension.
    
      The data is assumed to be a list of two numpy arrays.  The first element is
      the input data of shape (t, h, w, c_input).  The second element is the label
      data of shape (t, h, w, c_label).  I.e., all dimensions are the same, except
      the channel count.  NOTE: t is the length of an entire fire sequence, not the
      amount of time in a single temporal data point.
    
      See cfg.training.tree_noise_mean for description of noise.
    
      Args:
        data: The data.
        time_length: How many time steps should be in the returned data points.
        noise_mean: The mean parameter to use when adding noise.
        noise_std: The std parameter to use when adding noise.
        skip_initial_rows: The number of rows in the data input to skip.
    
      Yields:
        A 2-tuple containing
          [0] the input data sequence (t, h, w, c_input)
          [1] the label data sequence (t, h, w, c_label)
      """
      # data has shape 5 (s, t, h, w, c).  s is the number of samples, and then t is
      # the number of time points per sample.  To generate rows, we need to combine
      # the s and t channels together.
      assert len(data) == 2
      assert len(data[0].shape) == 4
    
      generator = tfHelpers.generate_rows(data, noise_mean, noise_std)
    
      time_steps = data[0].shape[0]
      # Build up time sequences from the input data.
      curr_row = 0
      while curr_row < time_steps:
        # Skip the initial rows.
        if curr_row < skip_initial_rows:
          curr_row += 1
          continue
    
        # Stop if we can't make a full sequence from this frame onward.
        if curr_row + time_length >= time_steps:
          break
        input_frames = []
        label_frames = []
        for _ in range(time_length):
          next_row = next(generator)
          input_frames.append(next_row[0])
          label_frames.append(next_row[1])
          curr_row += 1
    
        input_sequence = np.concatenate(input_frames, axis=0)
        label_sequence = np.concatenate(label_frames, axis=0)
        yield (input_sequence, label_sequence)
    
    
    def generate_time_instances(input_data: List[any],
                                noise_mean: float,
                                noise_std: float,
                                skip_initial_rows: int = -1):
      """Like generate_time_sequences, but returns one time point. (h, w, c)."""
      generator = tfHelpers.generate_rows(input_data, noise_mean, noise_std)
    
      # Return single data points without time.
      row_count = 0
      for row in generator:
        # Skip the initial rows.
        if row_count < skip_initial_rows:
          row_count += 1
        else:
          assert row[1].shape[0] == 1
          input_row = tf.reshape(row[0], shape=(row[0].shape)[1:])
          label_row = tf.reshape(row[1], shape=(row[1].shape)[1:])
          yield (input_row.numpy(), label_row.numpy())
