import jax.numpy as jnp
import jax.random as jr
import pandas as pd
import pickle
from dynamax.hidden_markov_model.models.arhmm import LinearAutoregressiveHMM
from dynamax.parameters import ParameterProperties
from sklearn.preprocessing import StandardScaler
import optax

def calculate_aic_bic(log_likelihood, num_params, num_obs):
    """Calculate AIC and BIC based on the log-likelihood, number of parameters, and observations."""
    aic = 2 * num_params - 2 * log_likelihood
    bic = num_params * jnp.log(num_obs) - 2 * log_likelihood
    return aic, bic

def fit_arhmm_and_compute_metrics(data, num_states, ar_order):
    # Remove zero-variance features
    data_reduced = remove_zero_variance_features(data)
    if data_reduced.size == 0:
        print("Data is empty after removing zero-variance features.")
        return jnp.nan, jnp.nan, jnp.nan

    # Normalize the data
    data_normalized = normalize_data(data_reduced)
    emissions = jnp.array(data_normalized)
    emission_dim = emissions.shape[1]

    # Check data shapes
    print(f"Emissions shape: {emissions.shape}")

    # Initialize the AR-HMM model
    arhmm = LinearAutoregressiveHMM(
        num_states=num_states,
        emission_dim=emission_dim,
        num_lags=ar_order
    )

    # Define 'key' before using it
    key = jr.PRNGKey(0)

    # Initialize parameters with K-means and set n_init
    params, param_props = arhmm.initialize(
        key,
        method="kmeans",
        emissions=emissions
    )

    # Compute inputs for the AR-HMM
    inputs = arhmm.compute_inputs(emissions)
    if inputs is None:
        print("Inputs is None, creating an empty array.")
        inputs = jnp.empty((emissions.shape[0], 0))  # Create an empty inputs array

    # Check inputs shape
    print(f"Inputs shape: {inputs.shape}")

    # Fit the model using SGD
    try:
        optimizer = optax.adam(learning_rate=1e-3)  # Adjust learning rate as needed
        num_epochs = 50  # Adjust the number of epochs as needed
        fitted_params, lps = arhmm.fit_sgd(
            params,
            param_props,
            emissions,
            inputs=inputs,
            num_epochs=num_epochs,
            optimizer=optimizer
        )
        # Check for NaNs in log-likelihood
        if jnp.isnan(lps).any():
            print("Log-likelihood contains NaNs.")
            return jnp.nan, jnp.nan, jnp.nan
    except Exception as e:
        print(f"Failed to fit model: {e}")
        return jnp.nan, jnp.nan, jnp.nan

    # Proceed with computing log-likelihood, AIC, BIC
    log_likelihood = lps[-1]

    # Number of parameters in the AR-HMM
    num_params = (
        num_states - 1  # initial state probabilities
        + num_states * (num_states - 1)  # transition matrix parameters
        + num_states * (ar_order * emission_dim * emission_dim)  # AR weights for emissions
        + num_states * emission_dim  # AR biases
        + num_states * emission_dim * (emission_dim + 1) // 2  # Covariance matrices
    )

    # Total number of observations (time steps)
    num_obs = data.shape[0]

    # Calculate AIC and BIC
    aic, bic = calculate_aic_bic(log_likelihood, num_params, num_obs)

    return aic, bic, log_likelihood

def grid_search_arhmm(datas, state_range, ar_order_range):
    """Perform a grid search to find the optimal number of states and AR order based on BIC."""
    best_bic = jnp.inf
    best_config = None

    for num_states in state_range:
        for ar_order in ar_order_range:
            print(f"\nTesting AR-HMM with {num_states} states and AR order {ar_order}")

            try:
                # Aggregate metrics over multiple datasets if provided
                total_aic, total_bic, total_ll = 0, 0, 0
                for data in datas:
                    aic, bic, log_likelihood = fit_arhmm_and_compute_metrics(data, num_states, ar_order)
                    total_aic += aic
                    total_bic += bic
                    total_ll += log_likelihood

                print(f"Total AIC: {total_aic}, Total BIC: {total_bic}, Total Log-Likelihood: {total_ll}")

                # Check for nan values
                if jnp.isnan(total_bic):
                    print(f"Encountered nan in metrics for {num_states} states, AR order {ar_order}")
                    continue  # Skip this configuration

                # Update the best model configuration if necessary
                if total_bic < best_bic:
                    best_bic = total_bic
                    best_config = (num_states, ar_order, total_aic, total_bic, total_ll)
                    print(f"New best config: {best_config}")

            except Exception as e:
                print(f"Failed for {num_states} states, AR order {ar_order}: {e}")

    print(f"\nBest config: {best_config}")
    return best_config

def remove_zero_variance_features(data):
    variances = data.var(axis=0)
    non_zero_variance_columns = variances > 1e-8  # Threshold to avoid numerical issues
    data_reduced = data[:, non_zero_variance_columns]
    return data_reduced

def normalize_data(data):
    scaler = StandardScaler()
    data_normalized = scaler.fit_transform(data)
    return data_normalized

def check_data_integrity(data):
    if jnp.isnan(data).any():
        print("Data contains NaNs.")
    if jnp.isinf(data).any():
        print("Data contains Infs.")
    if not jnp.isfinite(data).all():
        print("Data contains non-finite values.")
