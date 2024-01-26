# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 10:24:24 2024

@author: nikos
"""
import tensorflow as tf
from typing import Dict, Tuple, List, Text
import numpy as np
import helpers 
from helpers import Farsite2Google



def run_google_EPD_model(input):
    return tf.data.Dataset.from_tensor_slices(input)

def run_google_LSTM_model(input):
    return tf.data.Dataset.from_tensor_slices(input)
    
    
    