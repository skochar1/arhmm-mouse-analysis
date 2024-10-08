# Import necessary modules and functions
import jax.numpy as jnp
import pickle
from ..data_processing import load_and_preprocess_data, restructure_intervals_data
from grid_search_arhmm import grid_search_arhmm

# Define the path to your pickle files
pickle_files = ["path/to/file1.pkl", "path/to/file2.pkl"]  # Replace with actual file paths

# Step 1: Load and preprocess the data from pickle files
intervals_data_raw, feature_names = load_and_preprocess_data(pickle_files)

# Step 2: Restructure the data to match expected format for grid search
intervals_data = restructure_intervals_data(intervals_data_raw)

# Step 3: Define ranges for number of states and AR orders
state_range = range(55, 75)  # Testing number of states, modify as needed
ar_order_range = [1]  # Testing AR orders, modify as needed

# Step 4: Perform grid search for each interval
best_configs = {}
for interval_name, datas in intervals_data.items():
    print(f"\nProcessing {interval_name}...")
    best_config = grid_search_arhmm(datas, state_range, ar_order_range)
    best_configs[interval_name] = best_config
    print(f"Best configuration for {interval_name}: {best_config}")

# Step 5: Output the results
print("\nSummary of Best Configurations for All Intervals:")
for interval_name, config in best_configs.items():
    print(f"{interval_name}: States={config[0]}, AR Order={config[1]}, AIC={config[2]}, BIC={config[3]}, Log-Likelihood={config[4]}")
