import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from statsmodels.tsa.stattools import acf

def normalize_data(data):
    """
    Normalize the data using StandardScaler.

    Args:
    - data (np.ndarray): Input data array of shape (n_samples, n_features).

    Returns:
    - np.ndarray: Normalized data array.
    """
    scaler = StandardScaler()
    return scaler.fit_transform(data)

def perform_pca(data, num_pcs):
    """
    Perform Principal Component Analysis (PCA) and return the average of the top components.

    Args:
    - data (np.ndarray): Normalized data array of shape (n_samples, n_features).
    - num_pcs (int): Number of principal components to extract.

    Returns:
    - np.ndarray: Averaged principal components.
    """
    pca = PCA(n_components=num_pcs)
    principal_components = pca.fit_transform(data)
    return np.mean(principal_components, axis=1)

def shuffle_in_blocks(data, block_size):
    """
    Shuffle the data in specified block sizes.

    Args:
    - data (np.ndarray): 1D array of data to shuffle.
    - block_size (int): Number of samples per block.

    Returns:
    - np.ndarray: Data array with blocks shuffled.
    """
    num_samples = len(data)
    num_blocks = num_samples // block_size
    remaining_samples = num_samples % block_size

    # Extract and shuffle blocks
    blocks = np.array_split(data[:num_blocks * block_size], num_blocks)
    np.random.shuffle(blocks)

    # Concatenate shuffled blocks and any remaining samples
    shuffled_data = np.concatenate(blocks)
    if remaining_samples > 0:
        shuffled_data = np.concatenate((shuffled_data, data[-remaining_samples:]))

    return shuffled_data

def plot_autocorrelation(data, max_lag=1000, label=''):
    """
    Calculate and plot the autocorrelation of the data.

    Args:
    - data (np.ndarray): 1D array of data to analyze.
    - max_lag (int): Maximum number of lags to compute.
    - label (str): Label for the plot line.
    """
    acf_vals = acf(data, nlags=max_lag, fft=True)
    plt.plot(np.arange(len(acf_vals)), acf_vals, label=label)

def shuffling_analysis(training_data, num_pcs=9, block_sizes_ms=[100, 250, 500, 1000, 1500], fs=1000):
    """
    Perform shuffling and autocorrelation analysis on PCA-averaged data.

    Args:
    - training_data (list of np.ndarray): List of training data arrays to concatenate.
    - num_pcs (int): Number of principal components for PCA analysis.
    - block_sizes_ms (list of int): List of block sizes in milliseconds.
    - fs (int): Sampling rate in Hz for conversion from ms to samples.
    """
    # Step 1: Normalize the concatenated data
    all_training_data = np.concatenate(training_data, axis=0)
    data_normalized = normalize_data(all_training_data)

    # Step 2: Perform PCA and average the top principal components
    avg_pcs = perform_pca(data_normalized, num_pcs)

    # Step 3: Convert block sizes from ms to samples
    block_sizes_samples = [int((ms / 1000) * fs) for ms in block_sizes_ms]

    # Step 4: Perform block shuffling and autocorrelation analysis
    plt.figure(figsize=(12, 8))
    for block_size, block_size_ms in zip(block_sizes_samples, block_sizes_ms):
        data_shuffled = shuffle_in_blocks(avg_pcs, block_size)
        plot_autocorrelation(data_shuffled, label=f'{block_size_ms} ms blocks')

    # Step 5: Plot original data for comparison
    plot_autocorrelation(avg_pcs, label='Original Data', max_lag=1000)
    plt.xlabel('Time Lag (ms)')
    plt.ylabel('Autocorrelation')
    plt.title('Autocorrelation of Averaged PCs with Different Block Sizes')
    plt.legend()
    plt.show()
