# 1_data_availability.py
"""
Problem: Data Quality and Availability
Solution: Data imputation for missing values, basic noise reduction, and a conceptual example of data augmentation.

Description: Agricultural datasets often suffer from missing values, noise, and insufficient size.
This script demonstrates how to handle missing data using imputation, reduce noise with a simple filter,
and conceptually augment data (for numerical features).
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from scipy.signal import medfilt
import matplotlib.pyplot as plt # Added for potential future visualizations

def run_data_quality_solution(df_input=None):
    """
    Runs the data quality and availability solution.
    Args:
        df_input (pd.DataFrame, optional): Your actual agricultural data.
                                           If None, synthetic data will be generated.
    """
    print("--- Data Quality and Availability Solution ---")

    # --- Example Data ---
    if df_input is None:
        # Let's create a synthetic dataset with missing values and some noise
        np.random.seed(42)
        data = {
            'Temperature': np.random.normal(25, 5, 100),
            'Rainfall': np.random.normal(100, 30, 100),
            'Soil_Moisture': np.random.normal(0.6, 0.15, 100),
            'Yield': np.random.normal(500, 100, 100)
        }
        df = pd.DataFrame(data)

        # Introduce some missing values
        for col in ['Temperature', 'Rainfall', 'Soil_Moisture']:
            missing_indices = np.random.choice(df.index, size=10, replace=False)
            df.loc[missing_indices, col] = np.nan

        # Introduce some noise (e.g., sensor spikes)
        df['Temperature'] = df['Temperature'] + np.random.normal(0, 0.5, 100)
        df.loc[np.random.choice(df.index, size=5, replace=False), 'Temperature'] += 20 # Spikes
    else:
        df = df_input.copy()
        # Ensure the expected columns exist for imputation/filtering
        required_cols = ['Temperature', 'Rainfall', 'Soil_Moisture']
        if not all(col in df.columns for col in required_cols):
            print(f"Warning: Input DataFrame is missing one or more required columns for this script: {required_cols}. Using synthetic data instead.")
            return run_data_quality_solution(df_input=None) # Fallback to synthetic data
        print("Using provided DataFrame for Data Quality Solution.")


    print("Original Data (first 5 rows with potential issues):")
    print(df.head())
    print("\nMissing values before imputation:")
    print(df.isnull().sum())

    # --- Step 1: Handle Missing Values (Imputation) ---
    # Using mean imputation for numerical features
    imputer = SimpleImputer(strategy='mean')
    df_imputed = df.copy()
    # Apply imputation only to columns that might have missing values
    cols_to_impute = [col for col in ['Temperature', 'Rainfall', 'Soil_Moisture'] if col in df_imputed.columns]
    if cols_to_impute:
        df_imputed[cols_to_impute] = imputer.fit_transform(df_imputed[cols_to_impute])
    else:
        print("No columns found for imputation. Skipping imputation step.")


    print("\nData after Mean Imputation (first 5 rows):")
    print(df_imputed.head())
    print("\nMissing values after imputation:")
    print(df_imputed.isnull().sum())

    # --- Step 2: Noise Reduction (Simple Median Filter for Temperature) ---
    # Median filter is good for removing salt-and-pepper noise or spikes
    # Note: For time-series data, apply carefully. For general numerical features, it smooths.
    window_size = 5 # Must be odd
    df_cleaned = df_imputed.copy()
    if 'Temperature' in df_cleaned.columns:
        df_cleaned['Temperature_Filtered'] = medfilt(df_cleaned['Temperature'], kernel_size=window_size)
        print(f"\nTemperature before and after Median Filtering (window size={window_size}):")
        print(df_imputed[['Temperature']].head())
        print(df_cleaned[['Temperature_Filtered']].head())
    else:
        print("Temperature column not found for filtering. Skipping noise reduction step.")


    # --- Step 3: Conceptual Data Augmentation for Numerical Data ---
    # For numerical data, augmentation can involve adding small random noise,
    # or creating synthetic samples based on existing distributions.
    # This is a simple example: adding slight noise to create new data points.
    num_original_samples = len(df_cleaned)
    num_augmented_samples = 20 # Number of new samples to generate

    augmented_data = []
    for _ in range(num_augmented_samples):
        # Select a random original sample
        random_sample = df_cleaned.sample(1).iloc[0]
        new_sample = random_sample.copy()

        # Add small random noise to numerical features, checking if columns exist
        if 'Temperature_Filtered' in new_sample:
            new_sample['Temperature_Filtered'] += np.random.normal(0, 0.5)
        elif 'Temperature' in new_sample: # Fallback if Temperature_Filtered wasn't created
            new_sample['Temperature'] += np.random.normal(0, 0.5)

        if 'Rainfall' in new_sample:
            new_sample['Rainfall'] += np.random.normal(0, 2)
        if 'Soil_Moisture' in new_sample:
            new_sample['Soil_Moisture'] += np.random.normal(0, 0.01)
        if 'Yield' in new_sample:
            new_sample['Yield'] += np.random.normal(0, 5) # Augment target too, if appropriate for task

        augmented_data.append(new_sample)

    df_augmented = pd.concat([df_cleaned, pd.DataFrame(augmented_data)], ignore_index=True)

    print(f"\nOriginal number of samples: {num_original_samples}")
    print(f"Number of samples after augmentation: {len(df_augmented)}")
    print("Augmented Data (last 5 rows, showing new data points):")
    print(df_augmented.tail())

if __name__ == "__main__":
    # Example usage:
    # To run with synthetic data:
    run_data_quality_solution()

    # To run with your own data (uncomment and replace with your DataFrame):
    # your_data = pd.read_csv('your_crop_data.csv')
    # run_data_quality_solution(df_input=your_data)
