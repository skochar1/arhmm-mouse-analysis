import random
import jax.numpy as jnp
import jax.random as jr
import optax
from dynamax.hidden_markov_model.models.arhmm import LinearAutoregressiveHMM
from data_processing import load_and_preprocess_data, restructure_intervals_data
from grid_search_arhmm import remove_zero_variance_features, normalize_data

def load_and_split_data(intervals_data, interval_name, train_ratio=0.6, val_ratio=0.2):
    """Load, filter, shuffle, and split data into training, validation, and testing sets."""
    datas = intervals_data[interval_name]
    datas = [data for data in datas if data.size > 0]  # Filter empty arrays

    random.shuffle(datas)
    num_samples = len(datas)
    train_end = int(train_ratio * num_samples)
    val_end = int((train_ratio + val_ratio) * num_samples)
    
    training_data = datas[:train_end]
    validation_data = datas[train_end:val_end]
    testing_data = datas[val_end:]
    
    return training_data, validation_data, testing_data

def preprocess_data(datas):
    """Preprocess data by removing zero-variance features and normalizing."""
    preprocessed_data = [normalize_data(remove_zero_variance_features(data)) for data in datas]
    return preprocessed_data

def initialize_arhmm(emission_dim, num_states, num_lags):
    """Initialize the AR-HMM model with specified parameters."""
    arhmm = LinearAutoregressiveHMM(
        num_states=num_states,
        emission_dim=emission_dim,
        num_lags=num_lags
    )
    return arhmm

def train_arhmm(arhmm, training_data, num_states, num_lags, num_epochs=50, learning_rate=1e-3):
    """Train the AR-HMM model on the training data."""
    emissions = jnp.concatenate(training_data, axis=0)
    key = jr.PRNGKey(0)
    params, param_props = arhmm.initialize(key, method="kmeans", emissions=emissions)
    optimizer = optax.adam(learning_rate)
    
    for data in training_data:
        emissions = jnp.array(data)
        inputs = arhmm.compute_inputs(emissions)
        if inputs is None:
            inputs = jnp.empty((emissions.shape[0], 0))
        
        params, lps = arhmm.fit_sgd(
            params,
            param_props,
            emissions,
            inputs=inputs,
            num_epochs=num_epochs,
            optimizer=optimizer
        )
    
    return params

def evaluate_arhmm(arhmm, params, data):
    """Evaluate the AR-HMM model on a dataset and return log likelihood."""
    emissions = jnp.array(data)
    inputs = arhmm.compute_inputs(emissions)
    if inputs is None:
        inputs = jnp.empty((emissions.shape[0], 0))
    
    posterior = arhmm.filter(params, emissions, inputs=inputs)
    return posterior.marginal_loglik

def main(pickle_files, interval_name='interval_2', num_states=64, num_lags=1):
    # Load and preprocess data
    intervals_data_raw, feature_names = load_and_preprocess_data(pickle_files)
    intervals_data = restructure_intervals_data(intervals_data_raw)

    # Split data
    training_data, validation_data, testing_data = load_and_split_data(intervals_data, interval_name)

    # Preprocess each set
    training_data_preprocessed = preprocess_data(training_data)
    validation_data_preprocessed = preprocess_data(validation_data)
    testing_data_preprocessed = preprocess_data(testing_data)

    # Initialize AR-HMM model
    emission_dim = training_data_preprocessed[0].shape[1]
    arhmm = initialize_arhmm(emission_dim, num_states, num_lags)

    # Train the model
    params = train_arhmm(arhmm, training_data_preprocessed, num_states, num_lags)

    # Evaluate on validation and test sets
    print("\nEvaluating on Validation Data:")
    for val_data in validation_data_preprocessed:
        val_ll = evaluate_arhmm(arhmm, params, val_data)
        print(f"Validation Log Likelihood: {val_ll}")

    print("\nEvaluating on Test Data:")
    for test_data in testing_data_preprocessed:
        test_ll = evaluate_arhmm(arhmm, params, test_data)
        print(f"Test Log Likelihood: {test_ll}")

# Example usage
if __name__ == "__main__":
    pickle_files = ["path/to/file1.pkl", "path/to/file2.pkl"]  # Replace with actual paths
    main(pickle_files, interval_name='interval_2')
