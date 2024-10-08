# model_training.py
import jax.numpy as jnp
import jax.random as jr
from dynamax.hidden_markov_model.models.arhmm import LinearAutoregressiveHMM
import optax

def initialize_arhmm(num_states, emission_dim, num_lags):
    """
    Initializes an AR-HMM model with the specified parameters.
    """
    arhmm = LinearAutoregressiveHMM(num_states=num_states, emission_dim=emission_dim, num_lags=num_lags)
    return arhmm

def train_arhmm(arhmm, training_data, learning_rate=1e-3, num_epochs=50):
    """
    Trains the AR-HMM model on training data.
    """
    key = jr.PRNGKey(0)
    params, param_props = arhmm.initialize(key, method="kmeans", emissions=jnp.concatenate(training_data, axis=0))
    optimizer = optax.adam(learning_rate)
    for data in training_data:
        emissions = jnp.array(data)
        inputs = arhmm.compute_inputs(emissions) or jnp.empty((emissions.shape[0], 0))
        params, _ = arhmm.fit_sgd(params, param_props, emissions, inputs=inputs, num_epochs=num_epochs, optimizer=optimizer)
    return params

def evaluate_arhmm(arhmm, params, data_segments):
    """
    Evaluates the AR-HMM model on given data segments.
    """
    for data in data_segments:
        emissions = jnp.array(data)
        inputs = arhmm.compute_inputs(emissions) or jnp.empty((emissions.shape[0], 0))
        posterior = arhmm.filter(params, emissions, inputs=inputs)
        print(f"Log Likelihood: {posterior.marginal_loglik}")
