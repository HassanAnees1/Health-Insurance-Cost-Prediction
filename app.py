# Simple Health Insurance Cost Predictor - Gradio App
# Matches the exact output format of the original implementation

import gradio as gr
import pandas as pd
import joblib

# Load the saved model and artifacts
try:
    model_artifacts = joblib.load('insurance_cost_model.pkl')
    loaded_model = model_artifacts['model']
    loaded_scaler = model_artifacts['scaler']
    loaded_feature_columns = model_artifacts['feature_columns']
    loaded_model_name = model_artifacts['model_name']
    print("Model artifacts loaded successfully.")
except FileNotFoundError:
    print("Error: 'insurance_cost_model.pkl' not found. Please ensure the model was saved.")
    loaded_model = None
    loaded_scaler = None
    loaded_feature_columns = None
    loaded_model_name = None

# Define the prediction function for Gradio
def predict_charge(age, sex, bmi, children, smoker, region):
    if loaded_model is None:
        return "Error: Model not loaded."

    try:
        # Create input dataframe
        input_data = pd.DataFrame({
            'age': [age],
            'sex': [sex],
            'bmi': [bmi],
            'children': [children],
            'smoker': [smoker],
            'region': [region]
        })

        # Add age and BMI categories - ensure these match the preprocessing in the notebook
        input_data['age_group'] = pd.cut(input_data['age'],
                                       bins=[0, 25, 35, 50, 100],
                                       labels=['Young', 'Adult', 'Middle', 'Senior'],
                                       right=True, # Consistent with default pd.cut behavior
                                       include_lowest=True) # Ensure lowest age is included

        input_data['bmi_category'] = pd.cut(input_data['bmi'],
                                          bins=[0, 18.5, 25, 30, 100],
                                          labels=['Underweight', 'Normal', 'Overweight', 'Obese'],
                                          right=True,
                                          include_lowest=True)

        # One-hot encode
        # Need to handle potential missing categories if input data doesn't have all of them
        # A robust way is to fit the encoder on the original training data's categorical columns
        # or ensure the `pd.get_dummies` call creates columns for all possible categories
        # from the training data.

        # For simplicity here, we recreate the get_dummies process and align columns
        input_encoded = pd.get_dummies(input_data, columns=['sex', 'smoker', 'region', 'age_group', 'bmi_category'],
                                      prefix=['sex', 'smoker', 'region', 'age_group', 'bmi_cat'])

        # Ensure all columns from training data are present in the input
        # and reorder them to match the training data used by the model
        final_input = pd.DataFrame(0, index=[0], columns=loaded_feature_columns)
        for col in input_encoded.columns:
            if col in final_input.columns:
                final_input[col] = input_encoded[col].values # Use .values to avoid index issues

        # Make prediction
        if loaded_model_name in ['Support Vector Regression', 'Ridge Regression', 'Lasso Regression'] and loaded_scaler is not None:
            input_scaled = loaded_scaler.transform(final_input)
            prediction = loaded_model.predict(input_scaled)[0]
        else:
            prediction = loaded_model.predict(final_input)[0]

        return f"Predicted Insurance Charge: ${prediction:.2f}"

    except Exception as e:
        return f"An error occurred during prediction: {e}"


# Create Gradio Interface
if loaded_model is not None:
    interface = gr.Interface(
        fn=predict_charge,
        inputs=[
            gr.Slider(minimum=18, maximum=64, step=1, label="Age"),
            gr.Radio(choices=['female', 'male'], label="Sex"),
            gr.Number(label="BMI"),
            gr.Slider(minimum=0, maximum=5, step=1, label="Children"),
            gr.Radio(choices=['yes', 'no'], label="Smoker"),
            gr.Dropdown(choices=['northeast', 'southeast', 'southwest', 'northwest'], label="Region")
        ],
        outputs="text",
        title="Health Insurance Cost Predictor",
        description="Enter patient details to predict insurance charges."
    )

    # Launch the interface
    interface.launch()
else:
    print("Cannot launch Gradio interface because the model was not loaded.")