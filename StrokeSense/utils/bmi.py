import streamlit as st

def calculate_bmi(height_cm, weight_kg):
    height_m = height_cm / 100
    if height_m <= 0:
        raise ValueError("Height must be greater than 0 cm")
    if weight_kg <= 0:
        raise ValueError("Weight must be greater than 0 kg")
    # Calculate BMI
    bmi = round(weight_kg / (height_m ** 2), 2)
    # Display health messages in Streamlit
    if bmi < 18.5:
        st.warning(f"Your BMI is {bmi}. You are underweight. Please consult a healthcare provider for advice.")
    elif 18.5 <= bmi < 25:  # Normal weight range
        st.success(f"Your BMI is {bmi}. You have a normal weight. Keep maintaining a healthy lifestyle!")
    elif 25 <= bmi < 30:  # Overweight range
        st.warning(f"Your BMI is {bmi}. You are overweight. Consider lifestyle changes to improve your health.")
    elif bmi >= 30:  # Obese range
        st.error(f"Your BMI is {bmi}. You are obese. Please consult a healthcare provider for guidance.")
    # Determine BMI category
    if bmi < 25:
        bmi_category = "Normal weight"
    else:
        bmi_category = "Obese"

    return bmi, bmi_category
