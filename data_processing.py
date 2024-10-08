# data_processing.py
import random
import pandas as pd
import jax.numpy as jnp
import os

def load_and_preprocess_data(pickle_files):
    """
    Load data from a list of pickle files.
    Returns a raw intervals data object.
    """
    intervals_data_raw = []
    for file in pickle_files:
        with open(file, 'rb') as f:
            data = pd.read_pickle(f)
            intervals_data_raw.append(data)
    return intervals_data_raw

def restructure_intervals_data(intervals_data_raw):
    """
    Restructure the raw intervals data into a usable format.
    Returns restructured intervals data.
    """
    intervals_data = {}  # Add restructuring logic as needed
    # Code to restructure data as needed
    return intervals_data

def preprocess_data(data_segments, remove_zero_variance_features, normalize_data):
    """
    Preprocesses data segments by removing zero-variance features and normalizing.
    """
    preprocessed_data = [remove_zero_variance_features(data) for data in data_segments]
    preprocessed_data = [normalize_data(data) for data in preprocessed_data]
    return preprocessed_data

def split_data(data_segments):
    """
    Splits the data into training, validation, and testing sets.
    """
    random.shuffle(data_segments)
    num_samples = len(data_segments)
    train_end = int(0.6 * num_samples)
    val_end = int(0.8 * num_samples)
    training_data = data_segments[:train_end]
    validation_data = data_segments[train_end:val_end]
    testing_data = data_segments[val_end:]
    return training_data, validation_data, testing_data
