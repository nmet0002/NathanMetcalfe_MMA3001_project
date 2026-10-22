import pandas as pd
import numpy as np
def convert_data(mat_data):
    """
    Takes the raw MAT file loaded via SciPy and creates a filtered 
    pandas dataframe ready to analyse
    Inputs:
        mat_data: The scipy loaded MAT file (scipy.io.loadmat(file_path))
    Outputs:
        df_mill_filtered: Filtered pandas dataframe
    """


    # Access the single record containing all the data from mat_data['mill'][0]
    # This assumes mat_data['mill'] is a structured array of shape (1,)
    mill_data_struct = mat_data['mill'][0]

    # Get the field names from the structured data
    field_names = mill_data_struct.dtype.names

    # Initialize a list to hold records that will form the DataFrame
    records_for_df = []

    # Determine the number of primary records (e.g., number of 'case' entries).
    # We assume 'case' is a reliable field to get the overall length of primary records.
    num_primary_records = mill_data_struct['case'].flatten().shape[0]

    # Define fields that are expected to contain time-series arrays (to be stored as arrays in DataFrame cells)
    time_series_fields = ['vib_table', 'vib_spindle', 'AE_table', 'AE_spindle', 'smcAC', 'smcDC']

    for i in range(num_primary_records):
        current_record = {}
        for field in field_names:
            val = mill_data_struct[field][i]
            if field in time_series_fields:
                # For time-series fields, store the actual NumPy array for this record
                if isinstance(val, np.ndarray):
                    current_record[field] = val.flatten() # Store as 1D array
                else:
                    # If for some reason it's not an array, store as a single-element array
                    current_record[field] = np.array([val])
            else:
                # For scalar fields, extract the single value
                if isinstance(val, np.ndarray) and val.size == 1:
                    current_record[field] = val.item()
                elif isinstance(val, np.ndarray):
                    # If a 'scalar' field unexpectedly contains a multi-element array,
                    # convert it to list or decide how to handle (e.g., take first element)
                    current_record[field] = val.flatten().tolist()
                else:
                    current_record[field] = val
        records_for_df.append(current_record)

    # Create a Pandas DataFrame from the list of dictionaries
    df_mill = pd.DataFrame(records_for_df)

    # Drop rows where 'VB' is NaN
    df_mill_cleaned = df_mill.dropna(subset=['VB']).copy()

    #Drop cases that only contain a single entry
    case_counts = df_mill_cleaned.groupby('case').size()
    cases_to_remove = case_counts[case_counts == 1].index.tolist()
    df_mill_filtered = df_mill_cleaned[~df_mill_cleaned['case'].isin(cases_to_remove)].copy()

    return df_mill_filtered

def calculate_avg_magnitude(signal_array):
    """
    Takes the time series entries, calculates and returns
    the average magnitude
    Inputs: 
        signal_array: The time series numpy array
    Returns:
        np.mean(np.abs(arr)): The average magnitude of the array
    """
    arr = np.array(signal_array, dtype=float)
    if len(arr) == 0:
        return 0.0
    return np.mean(np.abs(arr))

def calculate_peak_magnitude(signal_array):
    """
    Takes the time series entries and finds the peak
    magnitude reached
    Inputs:
        signal_array: The time series numpy array
    Returns:
        np.max(np.abs(arr)): The peak magnitude of the array
    """
    arr = np.array(signal_array, dtype=float)
    if len(arr) == 0:
        return 0.0
    return np.max(np.abs(arr))
