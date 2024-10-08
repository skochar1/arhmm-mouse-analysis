import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from statsmodels.tsa.stattools import acf

def normalize_data(data):
    """
    Normalize the data using StandardScaler.

    Args:
    - data (np.ndarray): Input data array of shape (n_samples, n_features).

    Returns:
    - data_normalized (np.ndarray): Normalized data array.
    """
    scaler = StandardScaler()
    data_normalized = scaler.fit_transform(data)
    return data_normalized

def perform_pca(data, num_pcs):
    """
    Perform Principal Component Analysis (PCA) on the data.

    Args:
    - data (np.ndarray): Normalized data array of shape (n_samples, n_features).
    - num_pcs (int): Number of principal components to extract.

    Returns:
    - principal_components (np.ndarray): Transformed data with selected PCs.
    - variance_ratios (np.ndarray): Explained variance ratios of the selected PCs.
    - pca_components (np.ndarray): PCA components (feature loadings).
    """
    pca = PCA(n_components=num_pcs)
    principal_components = pca.fit_transform(data)
    variance_ratios = pca.explained_variance_ratio_
    pca_components = pca.components_
    return principal_components, variance_ratios, pca_components

def compute_average_autocorrelation(data, num_pcs, max_lag=1000):
    """
    Compute the average autocorrelation across multiple principal components.

    Args:
    - data (np.ndarray): PCA-transformed data of shape (n_samples, num_pcs).
    - num_pcs (int): Number of principal components to include in the average.
    - max_lag (int): Maximum number of lags for autocorrelation calculation.

    Returns:
    - avg_acf (np.ndarray): Average autocorrelation values across selected PCs.
    """
    acf_values = np.zeros(max_lag + 1)
    for i in range(num_pcs):
        pc_data = data[:, i]
        acf_vals = acf(pc_data, nlags=max_lag, fft=True)
        acf_values += acf_vals
    avg_acf = acf_values / num_pcs
    return avg_acf

def plot_autocorrelation(acf_values, title="Average Autocorrelation", label=""):
    """
    Plot the autocorrelation function.

    Args:
    - acf_values (np.ndarray): Autocorrelation values to plot.
    - title (str): Title of the plot.
    - label (str): Label for the plot line.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(acf_values, label=label)
    plt.xlabel('Lag (ms)')
    plt.ylabel('Autocorrelation')
    plt.title(title)
    plt.legend()
    plt.show()

def analyze_pca_components(pca_components, feature_names, num_pcs, top_n=3):
    """
    Analyze the PCA components and display top contributing features for each PC.

    Args:
    - pca_components (np.ndarray): PCA components (loadings) of shape (num_pcs, n_features).
    - feature_names (list): List of feature names corresponding to the original data.
    - num_pcs (int): Number of principal components analyzed.
    - top_n (int): Number of top contributing features to display per PC.
    
    Returns:
    - pca_df (pd.DataFrame): DataFrame of PCA components with feature names.
    """
    pca_df = pd.DataFrame(pca_components.T, columns=[f'PC {i+1}' for i in range(num_pcs)], index=feature_names)
    print(pca_df)
    
    for i in range(num_pcs):
        pc_loadings = pca_df[f'PC {i+1}'].abs()
        top_features = pc_loadings.nlargest(top_n).index.tolist()
        print(f'Top {top_n} features contributing to PC {i+1}: {top_features}')
    
    return pca_df
