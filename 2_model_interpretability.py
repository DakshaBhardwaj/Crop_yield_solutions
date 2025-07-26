# 2_model_interpretability.py
"""
Problem: Model Interpretability
Solution: Using SHAP (SHapley Additive exPlanations) to explain the predictions of a Machine Learning model.

Description: Complex AI models are often "black boxes." SHAP values help us understand how each feature
contributes to a model's prediction for a specific instance, making the model more transparent.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import shap 
import matplotlib.pyplot as plt

def run_model_interpretability_solution(df_input=None):
    """
    Runs the model interpretability solution using SHAP.
    Args:
        df_input (pd.DataFrame, optional): Your actual agricultural data.
                                           If None, synthetic data will be generated.
                                           Must contain 'Temperature', 'Rainfall', 'Soil_Moisture',
                                           'Fertilizer_Used', 'Sunlight_Hours' as features and 'Yield' as target.
    """
    print("--- Model Interpretability Solution (SHAP) ---")

    # --- Example Data ---
    if df_input is None:
        np.random.seed(43)
        data = {
            'Temperature': np.random.normal(25, 5, 200),
            'Rainfall': np.random.normal(100, 30, 200),
            'Soil_Moisture': np.random.normal(0.6, 0.15, 200),
            'Fertilizer_Used': np.random.normal(50, 10, 200),
            'Sunlight_Hours': np.random.normal(8, 2, 200),
            'Yield': (
                5 * np.random.normal(25, 5, 200) +
                0.5 * np.random.normal(100, 30, 200) +
                100 * np.random.normal(0.6, 0.15, 200) +
                2 * np.random.normal(50, 10, 200) +
                10 * np.random.normal(8, 2, 200) +
                np.random.normal(0, 20, 200) # Noise
            )
        }
        df_interpret = pd.DataFrame(data)
    else:
        df_interpret = df_input.copy()
        required_features = ['Temperature', 'Rainfall', 'Soil_Moisture', 'Fertilizer_Used', 'Sunlight_Hours']
        target_col = 'Yield'
        if not all(col in df_interpret.columns for col in required_features + [target_col]):
            print(f"Error: Input DataFrame is missing one or more required columns for this script: {required_features + [target_col]}. Please provide a DataFrame with these columns or run with synthetic data.")
            return
        print("Using provided DataFrame for Model Interpretability Solution.")


    # Define features (X) and target (y)
    X = df_interpret[['Temperature', 'Rainfall', 'Soil_Moisture', 'Fertilizer_Used', 'Sunlight_Hours']]
    y = df_interpret['Yield']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- Train a RandomForestRegressor model ---
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    print(f"\nModel R-squared on test set: {model.score(X_test, y_test):.2f}")

    # --- Use SHAP to explain model predictions ---
    # Create a SHAP explainer object
    explainer = shap.TreeExplainer(model)

    # Calculate SHAP values for the test set
    shap_values = explainer.shap_values(X_test)

    # --- Visualize SHAP results ---

    # Summary plot: Shows feature importance and impact across the dataset
    print("\nSHAP Summary Plot (Feature Importance and Direction of Impact):")
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.title("SHAP Feature Importance (Average Absolute SHAP Value)")
    plt.show()
    plt.close() # Close plot to free memory

    shap.summary_plot(shap_values, X_test, show=False)
    plt.title("SHAP Summary Plot (Impact and Direction)")
    plt.show()
    plt.close() # Close plot to free memory

    # Force plot: Explains a single prediction
    # Let's explain the first prediction in the test set
    print("\nSHAP Force Plot for a Single Prediction (first test instance):")
    # shap.initjs() # Initialize JavaScript for interactive plots in notebooks
    # Note: shap.force_plot generates an interactive JavaScript visualization.
    # It will render directly in environments that support JS output (like Jupyter/Colab).
    # If running this code from a plain console, this plot will not display interactively.
    # You might need to save it as an HTML file using shap.save_html() for viewing.
    shap.force_plot(explainer.expected_value, shap_values[0,:], X_test.iloc[0,:])


    print("\nInterpretation:")
    print("The SHAP summary plots show which features are most important for the model's predictions.")
    print("The bar plot shows the average impact magnitude of each feature.")
    print("The scatter plot shows how the value of a feature impacts the prediction (e.g., high temperature might increase yield).")
    print("The force plot for a single instance shows how each feature pushes the prediction away from the base value.")

if __name__ == "__main__":
    # Example usage:
    # To run with synthetic data:
    run_model_interpretability_solution()

    # To run with your own data (uncomment and replace with your DataFrame):
    # your_data = pd.read_csv('your_crop_data_for_interpretability.csv')
    # run_model_interpretability_solution(df_input=your_data)
