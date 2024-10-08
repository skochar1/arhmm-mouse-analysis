# autocorrelation_analysis.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from statsmodels.tsa.stattools import acf

def combine_and_normalize(data_segments):
    """
    Combines and normalizes the data segments.
    """
    all_data = np.concatenate(data_segments, axis=0)
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(all_data)
    return normalized_data

def perform_pca(data, num_components=5):
    """
    Performs PCA on the normalized data and returns the top principal components.
    """
    pca = PCA(n_components=num_components)
    principal_components = pca.fit_transform(data)
    variance_ratios = pca.explained_variance_ratio_
    return principal_components, variance_ratios

def compute_autocorrelation(data, max_lag=1000):
    """
    Computes the autocorrelation for a data sequence.
    """
    return acf(data, nlags=max_lag, fft=True)

def plot_autocorrelation(principal_components, max_lag=1000):
    """
    Plots the autocorrelation for the top principal components.
    """
    plt.figure(figsize=(12, 8))
    for i in range(principal_components.shape[1]):
        acf_vals = compute_autocorrelation(principal_components[:, i], max_lag)
        plt.plot(acf_vals, label=f'PC {i+1}')
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.title('Autocorrelation of Top Principal Components')
    plt.legend()
    plt.show()
