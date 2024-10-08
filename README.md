# AR-HMM Mouse Analysis

## Data Preparation

1. Place your data files in the designated directory. Update the `pickle_files` path in `main.py` to point to your data files.
2. Ensure your data files are in pickle format and structured as expected by `data_processing.py`.

## Running the Analysis

Clone the repository:
   ```bash
   git clone https://github.com/skochr1/arhmm-mouse-analysis.git
   cd arhmm-mouse-analysis
   ```

## Code Overview

### Data Processing

`data_processing.py` contains functions to load and preprocess data. Use this file to customize the data loading process if your data structure differs from the current format.

### Model Training

`arhmm_training.py` includes functions to initialize and train an AR-HMM model, suitable for sequence data analysis. The model can be configured by adjusting the number of states and lagged components.

### Changepoint Detection

`changepoint_analysis.py` provides a method to detect changepoints in the data using a filtered derivative algorithm. This is useful for identifying significant shifts in behavior patterns over time.

### Shuffling Behavior Analysis

`shuffling_analysis.py` performs PCA on normalized data, shuffles the averaged principal components in blocks of varying sizes, and analyzes the temporal structure by plotting autocorrelation functions for each block size

### Autocorrelation Analysis

`autocorrelation_analysis.py` applies PCA to the data and calculates the autocorrelation for the top principal components. This provides insights into repeating patterns and long-range dependencies in the data.

## Acknowledgments

This repository is part of research conducted by Shreya Kochar (advised by Christoph Gebhardt) as part of the Data Science Institute Scholars program at the Bendeskey Lab.
