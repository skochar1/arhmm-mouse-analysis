# data_processing.py
import pandas as pd

def load_and_preprocess_data(pickle_files):
    """
    Load and preprocess the data from pickle files, and retrieve feature names.

    Args:
    - pickle_files (list of str): List of file paths to pickle files.

    Returns:
    - intervals_data (dict): Dictionary where each key is a file identifier, 
      and each value is another dictionary containing intervals for that file.
    - feature_names (dict): Dictionary where each key is an interval name, 
      and each value is a list of feature names for that interval.
    """
    intervals_data = {}
    feature_names = {}  # Dictionary to store feature names for each interval

    for file in pickle_files:
        data = pd.read_pickle(file)

        # Use 'pup_homebase_dist_mm' to determine pup presence
        pup_present = ~data['pup_homebase_dist_mm'].isna()

        # Find indices where the pup is added and retrieved
        if pup_present.any():
            pup_added_idx = pup_present.idxmax()
            pup_retrieved_idx = pup_present[pup_present].index[-1]
            if pup_retrieved_idx is None or pd.isna(pup_retrieved_idx):
                pup_retrieved_idx = data.index[-1]
        else:
            pup_added_idx = data.index[-1] + 1
            pup_retrieved_idx = data.index[-1]

        # Define interaction parameter columns
        interaction_columns = [
            'pup_homebase_dist_mm',
            'pup_homebase_dist_change_mm_s^-1',
            'pup_adult_dist_mm',
            'pup_adult_dist_change_mm_s^-1',
            'pup_adult_head_dist_mm',
            'pup_adult_head_dist_change_mm_s^-1',
            'delta_body_ori_pup_deg',
            'delta_head_ori_pup_deg'
        ]

        # Split data into intervals and keep track of feature names
        interval_1 = data.loc[:pup_added_idx - 1].drop(columns=interaction_columns).dropna()
        interval_2 = data.loc[pup_added_idx:pup_retrieved_idx].dropna()
        interval_3 = data.loc[pup_retrieved_idx + 1:].drop(columns=interaction_columns).dropna()
        interval_4 = data.drop(columns=interaction_columns).dropna()

        # Convert DataFrames to NumPy arrays and capture feature names for each interval
        intervals = {
            'interval_1': interval_1.to_numpy(),
            'interval_2': interval_2.to_numpy(),
            'interval_3': interval_3.to_numpy(),
            'interval_4': interval_4.to_numpy()
        }
        
        # Store feature names for each interval if not already stored
        if 'interval_1' not in feature_names:
            feature_names['interval_1'] = interval_1.columns.tolist()
        if 'interval_2' not in feature_names:
            feature_names['interval_2'] = interval_2.columns.tolist()
        if 'interval_3' not in feature_names:
            feature_names['interval_3'] = interval_3.columns.tolist()
        if 'interval_4' not in feature_names:
            feature_names['interval_4'] = interval_4.columns.tolist()

        # Use the file name or a unique identifier as the key
        file_key = file  # Modify as needed to extract a unique identifier
        intervals_data[file_key] = intervals

    return intervals_data, feature_names

def restructure_intervals_data(intervals_data):
    """
    Restructure intervals_data to have interval names as keys and lists of data arrays as values.

    Args:
    - intervals_data (dict): Dictionary where each key is a file identifier, 
      and each value is a dictionary containing intervals for that file.

    Returns:
    - restructured_data (dict): Dictionary where each key is an interval name, 
      and each value is a list of data arrays corresponding to that interval across files.
    """
    restructured_data = {'interval_1': [], 'interval_2': [], 'interval_3': [], 'interval_4': []}
    for file_key, intervals in intervals_data.items():
        for interval_name, data_array in intervals.items():
            if data_array.size > 0:
                restructured_data[interval_name].append(data_array)
    return restructured_data
