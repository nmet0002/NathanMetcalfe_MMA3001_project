import pandas as pd
import numpy as np
from scipy.optimize import brentq
def convert_data(mat_data):
    """
    Takes the raw MAT file loaded via SciPy and creates a filtered 
    pandas dataframe ready to analyse
    
    Args:
        mat_data: The scipy loaded MAT file (scipy.io.loadmat(file_path))
        
    Returns:
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
    
    Args: 
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
    
    Args:
        signal_array: The time series numpy array
        
    Returns:
        np.max(np.abs(arr)): The peak magnitude of the array
    """
    arr = np.array(signal_array, dtype=float)
    if len(arr) == 0:
        return 0.0
    return np.max(np.abs(arr))

def predict_flank_wear(model):
    """
    Uses the trained Neural Network model to estimate
    flank wear in mm given certain parameters. Designed for
    smooth user experience, demonstration purposes
    
    Args:
        model: The trained MLP NN model
        
    Returns:
        predicted_vb: The estimated flank wear in mm
    """
    print("--- Mill Flank Wear (VB) Prediction Tool ---")

    # Validate DOC (Depth of Cut)
    while True:
        try:
            doc_val = float(input("Enter DOC (mm) [e.g., 0.75, 1.5]: "))
            if doc_val <= 0:
                print("DOC must be a positive number. Please try again.")
                continue
            break
        except ValueError:
            print("Invalid input. Please enter a numerical value for DOC.")

    # Validate Feed
    while True:
        try:
            feed_val = float(input("Enter Feed (mm/min) [e.g., 0.25, 0.5]: "))
            if feed_val <= 0:
                print("Feed must be a positive number. Please try again.")
                continue
            break
        except ValueError:
            print("Invalid input. Please enter a numerical value for Feed.")

    # Validate Material (1 or 2)
    while True:
        try:
            material_val = int(input("Enter Material (1 for Cast Iron, 2 for Steel): "))
            if material_val not in [1, 2]:
                print("Material must be exactly 1 (Cast Iron) or 2 (Steel). Please try again.")
                continue
            break
        except ValueError:
            print("Invalid input. Please enter 1 or 2.")

    # Validate Time
    while True:
        try:
            time_val = float(input("Enter Time (minutes): "))
            if time_val < 0:
                print("Time cannot be negative. Please try again.")
                continue
            break
        except ValueError:
            print("Invalid input. Please enter a numerical value for Time.")

    # Prepare features as a pandas DataFrame with matching feature names to avoid the UserWarning
    input_data = pd.DataFrame([[time_val, doc_val, feed_val, material_val]],
                              columns=['time', 'DOC', 'feed', 'material'])

    # Predict using the trained MLP pipeline
    predicted_vb = model.predict(input_data)[0]

    print("\n--- Prediction Results ---")
    print(f"Inputs -> Time: {time_val} min | DOC: {doc_val} mm | Feed: {feed_val} | Material: {'Cast Iron (1)' if material_val == 1 else 'Steel (2)'}")
    print(f"Predicted Flank Wear (VB): {predicted_vb:.4f} mm")
    return predicted_vb


def predict_time_to_threshold(model):
    """
    Uses root finding methods to estimate the time when a certain
    threshold flank wear will be reached.

    Args:
        model: The trained NN MLP model

    Returns:
        target_time: The time at which threshold VB is estimated to be reached, in minutes
    """
    
    print("--- Time to Flank Wear (VB) Threshold Tool ---")

    # Validate Threshold VB
    while True:
        try:
            threshold_vb = float(input("Enter target VB threshold (mm) [e.g., 0.3, 0.5]: "))
            if threshold_vb <= 0:
                print("Threshold VB must be positive. Please try again.")
                continue
            break
        except ValueError:
            print("Invalid input. Please enter a numerical value for VB.")

    # Validate DOC (Depth of Cut)
    while True:
        try:
            doc_val = float(input("Enter DOC (mm) [e.g., 0.75, 1.5]: "))
            if doc_val <= 0:
                print("DOC must be a positive number. Please try again.")
                continue
            break
        except ValueError:
            print("Invalid input. Please enter a numerical value for DOC.")

    # Validate Feed
    while True:
        try:
            feed_val = float(input("Enter Feed (mm/min) [e.g., 0.25, 0.5]: "))
            if feed_val <= 0:
                print("Feed must be a positive number. Please try again.")
                continue
            break
        except ValueError:
            print("Invalid input. Please enter a numerical value for Feed.")

    # Validate Material (1 or 2)
    while True:
        try:
            material_val = int(input("Enter Material (1 for Cast Iron, 2 for Steel): "))
            if material_val not in [1, 2]:
                print("Material must be exactly 1 (Cast Iron) or 2 (Steel). Please try again.")
                continue
            break
        except ValueError:
            print("Invalid input. Please enter 1 or 2.")

    # Define the objective function for root-finding:
    # We want to find a time 't' where model_prediction(t) - threshold_vb = 0
    def objective_function(t):
        input_data = pd.DataFrame([[t, doc_val, feed_val, material_val]],
                                  columns=['time', 'DOC', 'feed', 'material'])
        return model.predict(input_data)[0] - threshold_vb

    # Establish bounds for finding the root (Time from 0 to 200 minutes)
    t_min, t_max = 0.0, 200.0

    try:
        # Check if a root exists in the given interval
        f_min = objective_function(t_min)
        f_max = objective_function(t_max)

        if f_min * f_max > 0:
            if f_min > 0:
                print(f"\n[Notice] Even at t = 0 min, wear ({f_min + threshold_vb:.4f} mm) is already above the threshold of {threshold_vb} mm.")
            else:
                print(f"\n[Notice] The threshold of {threshold_vb} mm is not reached within the maximum safe range of {t_max} minutes.")
        else:
            # Solve for time where prediction equals threshold
            target_time = brentq(objective_function, t_min, t_max)
            print("\n--- Estimation Results ---")
            print(f"Inputs -> Target VB: {threshold_vb} mm | DOC: {doc_val} mm | Feed: {feed_val} | Material: {'Cast Iron (1)' if material_val == 1 else 'Steel (2)'}")
            print(f"Estimated Time to reach threshold: {target_time:.2f} minutes")
            return target_time

    except Exception as e:
        print(f"An error occurred during root-finding search: {e}")
