# changepoint_detection.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

def filtered_derivative_algorithm(random_projections, k=4, h=0.15, sigma=0.43, crossing_window=5):
    """
    Apply a filtered derivative algorithm to detect changepoints.
    """
    unit_normalized = random_projections / np.linalg.norm(random_projections, axis=1, keepdims=True)
    derivatives = np.diff(unit_normalized, axis=0)
    lagged_derivatives = derivatives[k:]
    binarized_signal = np.sum(lagged_derivatives > h, axis=1)
    smoothed_signal = gaussian_filter1d(binarized_signal, sigma=sigma)
    raw_changepoints, _ = find_peaks(smoothed_signal)
    confirmed_changepoints = []
    for point in raw_changepoints:
        if point >= crossing_window and point < len(smoothed_signal) - crossing_window:
            crossing_count = np.sum(smoothed_signal[point - crossing_window:point + crossing_window] > h)
            if crossing_count > crossing_window // 2:
                confirmed_changepoints.append(point)
    return confirmed_changepoints

def plot_changepoints(data, changepoints):
    """
    Plot the data with detected changepoints marked.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(np.mean(data, axis=1), label='Data')
    for cp in changepoints:
        plt.axvline(cp, color='red', linestyle='--', alpha=0.5)
    plt.xlabel('Time')
    plt.ylabel('Data Value')
    plt.title('Detected Changepoints')
    plt.legend()
    plt.show()
