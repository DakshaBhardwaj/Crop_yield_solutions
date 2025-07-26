# optimal_harvest_scheduling.py
"""
Problem: Optimal Harvest Scheduling
Solution: Use a Machine Learning model to predict optimal harvest windows based on historical market data and environmental factors.

Description: Traditional harvest timing relies on intuition, leading to potential spoilage or lower quality.
This model uses historical data to predict the best time to harvest for maximum profitability.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns # Not directly used for plotting in this version, but often useful for data analysis

def run_optimal_harvest_scheduling_solution(df_input=None, fixed_conditions_input=None):
    """
    Runs the optimal harvest scheduling solution.
    Args:
        df_input (pd.DataFrame, optional): Your actual historical harvest data.
                                           If None, synthetic data will be generated.
                                           Must contain 'Avg_Temp_PreHarvest', 'Total_Rain_PreHarvest',
                                           'Sunlight_Hours_PreHarvest', 'Days_Since_Planting',
                                           'Market_Demand_Index' as features and 'Crop_Quality_Score' as target.
        fixed_conditions_input (dict, optional): A dictionary of fixed environmental/market conditions
                                                  for simulation. If None, default values are used.
    """
    print("--- Optimal Harvest Scheduling Solution ---")

    # --- Example Data ---
    if df_input is None:
        # Synthetic dataset for harvest scheduling
        # Features: Temperature, Rainfall, Sunlight, Days_Since_Planting, Market_Demand_Index
        # Target: Crop_Quality_Score (higher is better for harvest)
        np.random.seed(45)
        num_samples = 300
        data_harvest = {
            'Avg_Temp_PreHarvest': np.random.normal(22, 3, num_samples), # Avg temp in weeks before harvest
            'Total_Rain_PreHarvest': np.random.normal(80, 20, num_samples), # Total rain in weeks before harvest
            'Sunlight_Hours_PreHarvest': np.random.normal(7, 1.5, num_samples), # Avg sunlight in weeks before harvest
            'Days_Since_Planting': np.random.randint(90, 180, num_samples), # Days from planting to harvest attempt
            'Market_Demand_Index': np.random.uniform(0.5, 1.5, num_samples), # Index of market demand (1.0 is average)
            'Crop_Quality_Score': np.zeros(num_samples)
        }
        df_harvest = pd.DataFrame(data_harvest)

        # Simulate crop quality based on features (simplified relationship)
        # Optimal quality around certain temperature, rainfall, and specific planting days
        df_harvest['Crop_Quality_Score'] = (
            50 +
            (df_harvest['Avg_Temp_PreHarvest'] - 22)**2 * -2 + # Quadratic effect, optimal around 22
            (df_harvest['Total_Rain_PreHarvest'] - 80)**2 * -0.1 + # Quadratic effect, optimal around 80
            df_harvest['Sunlight_Hours_PreHarvest'] * 5 +
            (df_harvest['Days_Since_Planting'] - 135)**2 * -0.05 + # Optimal around 135 days
            df_harvest['Market_Demand_Index'] * 20 +
            np.random.normal(0, 5, num_samples) # Noise
        )
        df_harvest['Crop_Quality_Score'] = np.clip(df_harvest['Crop_Quality_Score'], 0, 100) # Clip to 0-100
    else:
        df_harvest = df_input.copy()
        required_cols = [
            'Avg_Temp_PreHarvest', 'Total_Rain_PreHarvest', 'Sunlight_Hours_PreHarvest',
            'Days_Since_Planting', 'Market_Demand_Index', 'Crop_Quality_Score'
        ]
        if not all(col in df_harvest.columns for col in required_cols):
            print(f"Error: Input DataFrame is missing one or more required columns for this script: {required_cols}. Please provide a DataFrame with these columns or run with synthetic data.")
            return
        print("Using provided DataFrame for Optimal Harvest Scheduling.")


    print("Sample Harvest Data (first 5 rows):")
    print(df_harvest.head())

    # Define features (X) and target (y)
    X_harvest = df_harvest.drop('Crop_Quality_Score', axis=1)
    y_harvest = df_harvest['Crop_Quality_Score']

    # Split data
    X_train_harvest, X_test_harvest, y_train_harvest, y_test_harvest = train_test_split(
        X_harvest, y_harvest, test_size=0.2, random_state=42
    )

    # --- Train a RandomForestRegressor model ---
    model_harvest = RandomForestRegressor(n_estimators=100, random_state=42)
    model_harvest.fit(X_train_harvest, y_train_harvest)

    # Evaluate the model
    y_pred_harvest = model_harvest.predict(X_test_harvest)
    mae = mean_absolute_error(y_test_harvest, y_pred_harvest)
    r2 = r2_score(y_test_harvest, y_pred_harvest)

    print(f"\nModel MAE: {mae:.2f}")
    print(f"Model R-squared: {r2:.2f}")

    # --- Predict Optimal Harvest Window ---
    # Let's simulate different 'Days_Since_Planting' for a fixed set of environmental conditions
    # to find the optimal harvest window.
    if fixed_conditions_input is None:
        fixed_conditions = {
            'Avg_Temp_PreHarvest': 22,
            'Total_Rain_PreHarvest': 80,
            'Sunlight_Hours_PreHarvest': 7,
            'Market_Demand_Index': 1.2 # Assuming high demand
        }
    else:
        fixed_conditions = fixed_conditions_input
        expected_keys = ['Avg_Temp_PreHarvest', 'Total_Rain_PreHarvest', 'Sunlight_Hours_PreHarvest', 'Market_Demand_Index']
        if not all(key in fixed_conditions for key in expected_keys):
            print(f"Warning: fixed_conditions_input is missing one or more required keys: {expected_keys}. Using default fixed conditions.")
            fixed_conditions = {
                'Avg_Temp_PreHarvest': 22,
                'Total_Rain_PreHarvest': 80,
                'Sunlight_Hours_PreHarvest': 7,
                'Market_Demand_Index': 1.2
            }


    days_range = np.arange(90, 181, 1) # Test days from 90 to 180
    predictions_quality = []

    for days in days_range:
        test_data = pd.DataFrame([[
            fixed_conditions['Avg_Temp_PreHarvest'],
            fixed_conditions['Total_Rain_PreHarvest'],
            fixed_conditions['Sunlight_Hours_PreHarvest'],
            days,
            fixed_conditions['Market_Demand_Index']
        ]], columns=X_harvest.columns)
        predicted_quality = model_harvest.predict(test_data)[0]
        predictions_quality.append(predicted_quality)

    # Find the day with the highest predicted quality
    optimal_day_index = np.argmax(predictions_quality)
    optimal_days_since_planting = days_range[optimal_day_index]
    max_predicted_quality = predictions_quality[optimal_day_index]

    print(f"\nPredicted optimal harvest window (days since planting): {optimal_days_since_planting} days")
    print(f"Maximum predicted crop quality score: {max_predicted_quality:.2f}")

    # Visualize the predicted quality over time
    plt.figure(figsize=(10, 6))
    plt.plot(days_range, predictions_quality, marker='o', linestyle='-', color='green', markersize=4)
    plt.axvline(x=optimal_days_since_planting, color='red', linestyle='--', label=f'Optimal Harvest Day: {optimal_days_since_planting}')
    plt.title('Predicted Crop Quality vs. Days Since Planting')
    plt.xlabel('Days Since Planting')
    plt.ylabel('Predicted Crop Quality Score (0-100)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.close() # Close plot to free memory

    print("\nInterpretation:")
    print("The plot shows how the predicted crop quality changes based on the 'Days Since Planting'.")
    print("The red dashed line indicates the optimal harvest day for maximum quality under the given environmental and market conditions.")
    print("This allows farmers to make data-driven decisions for harvest timing to maximize quality and profitability.")

if __name__ == "__main__":
    # Example usage:
    # To run with synthetic data:
    run_optimal_harvest_scheduling_solution()

    # To run with your own data and specific fixed conditions (uncomment and replace):
    # your_harvest_data = pd.read_csv('your_harvest_data.csv')
    # custom_fixed_conditions = {
    #     'Avg_Temp_PreHarvest': 25,
    #     'Total_Rain_PreHarvest': 70,
    #     'Sunlight_Hours_PreHarvest': 8,
    #     'Market_Demand_Index': 1.0
    # }
    # run_optimal_harvest_scheduling_solution(df_input=your_harvest_data, fixed_conditions_input=custom_fixed_conditions)
