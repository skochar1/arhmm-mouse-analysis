# Documentation for the AR-HMM Grid Search Code

This code performs a grid search to find the optimal number of hidden states and autoregressive (AR) order for an Autoregressive Hidden Markov Model (AR-HMM) using behavioral data.

**Overview:**

- **Objective:** Identify the best AR-HMM configuration that models the given data by varying the number of hidden states and AR orders.
- **Approach:** Load and preprocess the data, then perform a grid search over specified ranges of hidden states and AR orders. For each configuration, fit an AR-HMM using Stochastic Gradient Descent (SGD) and compute model selection metrics (AIC and BIC) to evaluate performance.

**Why Use Stochastic Gradient Descent (SGD)?**

- **Robustness:** The EM algorithm initially used for fitting encountered numerical instability and `NaN` values in the log-likelihood.
- **SGD Benefits:** SGD updates model parameters incrementally and is more robust to such issues, providing stable convergence and valid log-likelihood values.
- **Improved Fitting:** By using SGD, we can effectively fit the AR-HMM to our data, even when the EM algorithm fails.

**Key Components of the Code:**

1. **Imports:**

   - **JAX Libraries:** For high-performance numerical computations and random number generation.
   - **Pandas and Pickle:** To load and handle data stored in pickle files.
   - **dynamax Library:** Provides the AR-HMM model implementation.
   - **scikit-learn's StandardScaler:** For normalizing the data.
   - **optax:** Optimization library used for implementing SGD.

2. **Helper Functions:**

   - **`calculate_aic_bic(log_likelihood, num_params, num_obs)`:**
     - Calculates the Akaike Information Criterion (AIC) and Bayesian Information Criterion (BIC) to evaluate model performance, balancing model fit and complexity.
   - **`fit_arhmm_and_compute_metrics(data, num_states, ar_order)`:**
     - Fits an AR-HMM to the data using SGD and computes AIC, BIC, and log-likelihood for the model.
   - **`grid_search_arhmm(datas, state_range, ar_order_range)`:**
     - Performs a grid search over specified ranges of hidden states and AR orders to find the optimal model configuration based on BIC.
   - **`remove_zero_variance_features(data)`:**
     - Removes features that have zero variance, which can cause numerical issues during model fitting.
   - **`normalize_data(data)`:**
     - Normalizes the data to have zero mean and unit variance, aiding in the stability and convergence of the model fitting process.
   - **`check_data_integrity(data)`:**
     - Checks the data for any `NaN` or infinite values that could disrupt model fitting.

3. **Data Loading and Preprocessing:**

   - **`intervals_data = load_and_preprocess_data(pickle_files)`:**
     - Loads data from pickle files and preprocesses it for modeling. Note that `load_and_preprocess_data` should be defined to handle your specific data structure.

4. **Grid Search Execution:**

   - **Parameter Ranges:**
     - **`state_range`:** Range of hidden states to try (e.g., from 2 to 3).
     - **`ar_order_range`:** Range of AR orders to try (e.g., from 1 to 2).
   - **Iteration:**
     - For each interval in the data, the code performs the grid search using `grid_search_arhmm` to find the best model configuration.

**How the Grid Search Works:**

- The grid search systematically tests combinations of hidden states and AR orders.
- For each combination:
  - The model is fitted to the data using SGD.
  - The log-likelihood of the fitted model is obtained.
  - AIC and BIC are calculated to assess model performance while penalizing for complexity.
- The best configuration is selected based on the lowest BIC value.

**Function Explanations:**

- **`calculate_aic_bic`:**
  - **Purpose:** Computes AIC and BIC to evaluate and compare different models.
  - **Inputs:**
    - `log_likelihood`: The log-likelihood from the fitted model.
    - `num_params`: Total number of parameters estimated in the model.
    - `num_obs`: Number of observations (data points) in the dataset.
  - **Outputs:**
    - `aic`: The AIC value.
    - `bic`: The BIC value.

- **`fit_arhmm_and_compute_metrics`:**
  - **Purpose:** Fits the AR-HMM using SGD and computes model evaluation metrics.
  - **Inputs:**
    - `data`: The dataset to fit the model to.
    - `num_states`: Number of hidden states to use in the model.
    - `ar_order`: The order of the autoregressive component.
  - **Process:**
    1. **Preprocessing:**
       - Removes features with zero variance.
       - Normalizes the data.
    2. **Model Initialization:**
       - Sets up the AR-HMM with the specified parameters.
    3. **Parameter Initialization:**
       - Initializes model parameters using K-means clustering.
    4. **Model Fitting:**
       - Uses SGD to fit the model to the data.
       - Checks for convergence issues like `NaN` values in the log-likelihood.
    5. **Metric Calculation:**
       - Calculates AIC, BIC, and the final log-likelihood.

- **`grid_search_arhmm`:**
  - **Purpose:** Finds the optimal number of hidden states and AR order by comparing models based on BIC.
  - **Inputs:**
    - `datas`: A list of datasets to fit the model on (e.g., different time intervals).
    - `state_range`: A range of hidden state numbers to try.
    - `ar_order_range`: A range of AR orders to try.
  - **Process:**
    - Iterates over all combinations of `num_states` and `ar_order`.
    - Aggregates the metrics over all datasets.
    - Updates the best configuration when a model with a lower BIC is found.

- **`remove_zero_variance_features`:**
  - **Purpose:** Prevents numerical issues by removing features that do not vary across observations.
  - **Input:** The dataset to process.
  - **Output:** The dataset with zero-variance features removed.

- **`normalize_data`:**
  - **Purpose:** Standardizes features to improve model convergence.
  - **Input:** The dataset to normalize.
  - **Output:** The normalized dataset.

- **`check_data_integrity`:**
  - **Purpose:** Ensures the dataset is free from `NaN` or infinite values before model fitting.
  - **Input:** The dataset to check.
  - **Output:** Prints messages if issues are found.

**Conclusion:**

- **Goal Achieved:** By using SGD, the code effectively fits AR-HMMs to the data, even in situations where traditional methods like EM fail.
- **Model Selection:** The grid search identifies the best model configuration by balancing model fit and complexity, as indicated by the BIC metric.
- **Applicability:** This approach can be used to uncover hidden patterns or states in behavioral data, aiding in further analysis or research.
