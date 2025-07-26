# 3_cost_accessibility.py
"""
Problem: Accessibility & Cost
Solution: Demonstrate a lightweight model suitable for edge devices and a conceptual function for sending SMS alerts to farmers.

Description: High implementation costs and infrastructure gaps limit AI adoption.
Lightweight models reduce computational burden, and SMS alerts provide crucial information
without requiring internet access or smartphones.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression # A very lightweight model
import time
import matplotlib.pyplot as plt # Added for potential future visualizations

def send_sms_alert(phone_number, message):
    """
    Simulates sending an SMS alert to a farmer.
    In a real application, this would use an SMS gateway API (e.g., Twilio, Nexmo).
    This function is a simulation for demonstration purposes and does not send actual SMS.
    """
    print(f"\n--- SIMULATED SMS ALERT ---")
    print(f"Sending SMS to {phone_number}:")
    print(f"Message: '{message}'")
    print(f"--------------------------")
    # Example of how a real API call might look (conceptual, requires API keys and library):
    # from twilio.rest import Client
    # TWILIO_ACCOUNT_SID = 'ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxx' # Replace with your Twilio Account SID
    # TWILIO_AUTH_TOKEN = 'your_auth_token' # Replace with your Twilio Auth Token
    # TWILIO_PHONE_NUMBER = '+15017122661' # Replace with your Twilio phone number
    # try:
    #     client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    #     message = client.messages.create(
    #         to=phone_number,
    #         from_=TWILIO_PHONE_NUMBER,
    #         body=message
    #     )
    #     print(f"SMS sent successfully! SID: {message.sid}")
    # except Exception as e:
    #     print(f"Error sending SMS: {e}")

def run_accessibility_cost_solution(df_input=None, farmer_phone_number="+919876543210"):
    """
    Runs the accessibility and cost solution, demonstrating a lightweight model
    and a conceptual SMS alert system.
    Args:
        df_input (pd.DataFrame, optional): Your actual agricultural data.
                                           If None, synthetic data will be generated.
                                           Must contain 'Temperature', 'Rainfall', 'Soil_Moisture'
                                           as features and 'Yield' as target.
        farmer_phone_number (str): The phone number to send simulated SMS alerts to.
    """
    print("--- Accessibility & Cost Solution ---")

    # --- Part 1: Lightweight Model Demonstration ---
    # Using a simple Linear Regression model as an example of a lightweight model
    # that requires less computation than deep learning models.

    # --- Example Data ---
    if df_input is None:
        np.random.seed(44)
        data_light = {
            'Temperature': np.random.normal(25, 5, 100),
            'Rainfall': np.random.normal(100, 30, 100),
            'Soil_Moisture': np.random.normal(0.6, 0.15, 100),
            'Yield': (
                2 * np.random.normal(25, 5, 100) +
                0.3 * np.random.normal(100, 30, 100) +
                50 * np.random.normal(0.6, 0.15, 100) +
                np.random.normal(0, 10, 100)
            )
        }
        df_light = pd.DataFrame(data_light)
    else:
        df_light = df_input.copy()
        required_features = ['Temperature', 'Rainfall', 'Soil_Moisture']
        target_col = 'Yield'
        if not all(col in df_light.columns for col in required_features + [target_col]):
            print(f"Error: Input DataFrame is missing one or more required columns for the lightweight model: {required_features + [target_col]}. Please provide a DataFrame with these columns or run with synthetic data.")
            return
        print("Using provided DataFrame for Lightweight Model.")


    X_light = df_light[['Temperature', 'Rainfall', 'Soil_Moisture']]
    y_light = df_light['Yield']

    X_train_light, X_test_light, y_train_light, y_test_light = train_test_split(
        X_light, y_light, test_size=0.2, random_state=42
    )

    # Train a lightweight model (Linear Regression)
    print("\nTraining a Lightweight Linear Regression Model...")
    start_time = time.time()
    light_model = LinearRegression()
    light_model.fit(X_train_light, y_train_light)
    end_time = time.time()

    print(f"Lightweight model training time: {end_time - start_time:.4f} seconds")
    print(f"Lightweight model R-squared on test set: {light_model.score(X_test_light, y_test_light):.2f}")

    # Simulate a prediction on an edge device
    new_data_point = pd.DataFrame([[28, 95, 0.65]], columns=['Temperature', 'Rainfall', 'Soil_Moisture'])
    prediction = light_model.predict(new_data_point)
    print(f"Lightweight model prediction for new data [Temp:28, Rain:95, Soil:0.65]: {prediction[0]:.2f} units")
    print("\nThis model is much faster to train and predict, suitable for deployment on low-power devices.")

    # --- Part 2: Conceptual SMS Alert Function ---
    yield_forecast_message = f"Your predicted crop yield for next season is {prediction[0]:.2f} units. Consider adjusting irrigation."
    send_sms_alert(farmer_phone_number, yield_forecast_message)

    disease_alert_message = "Urgent: Signs of blight detected in Sector C. Apply fungicide immediately."
    send_sms_alert(farmer_phone_number, disease_alert_message)

if __name__ == "__main__":
    # Example usage:
    # To run with synthetic data:
    run_accessibility_cost_solution()

    # To run with your own data and a specific phone number (uncomment and replace):
    # your_data = pd.read_csv('your_crop_data_for_lightweight_model.csv')
    # run_accessibility_cost_solution(df_input=your_data, farmer_phone_number="+1234567890")
