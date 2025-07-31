import streamlit as st
from utils.bmi import calculate_bmi
from utils.risk_n_level import age_gender_to_risk, age_to_age_group, health_risk_level, stress_level_category
from config.theme import theme, app_background
from config.design import input_design, hero_section, footer
import os

# ==========================
# PAGE CONFIGURATION
# ==========================
st.set_page_config(
    page_title="StrokeSense - Input Form",
    page_icon= os.path.join("..", "assets", "icon.png"),
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply the theme, background animation, and input design styles
theme()
app_background()
input_design()

# ==========================
# PAGE TITLE AND HERO SECTION
# ==========================
hero_section()
st.markdown("""
    <div class="hero-container">
        <h1>Stroke Risk Prediction – Input Form</h1>
        <p>Fill in your details below and click <b>Predict</b> to estimate your risk.</p>
    </div>
""", unsafe_allow_html=True)

# ==========================
# PERSONAL INFORMATION
# ==========================
with st.expander('Personal Information', expanded=True):
    # Collect user inputs for personal details
    age = st.number_input("Age (years)", 0, 100, 30, step=1, help="Enter your age in years.")
    gender = st.selectbox("Gender", ["Male", "Female"])
    height = st.number_input("Height (cm)", 100, 250, 170)
    weight = st.number_input("Weight (kg)", 30, 200, 70)
    
    # Calculate BMI and display it
    bmi, bmi_category = calculate_bmi(height, weight)
    st.write(f"**BMI:** {bmi}")  # Show only the raw BMI to the user

# ==========================
# MEDICAL HISTORY
# ==========================
with st.expander('Medical History', expanded=True):
    # Collect user inputs for medical history
    hypertension = st.selectbox("High Blood Pressure", ["No", "Yes"])
    st.markdown("<p class='field-desc'>Do you have high blood pressure?</p>", unsafe_allow_html=True)
    heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])
    st.markdown("<p class='field-desc'>Do you have any heart disease? Such as heart attack or angina.</p>", unsafe_allow_html=True)
    diabetes = st.selectbox("Diabetes", ["No", "Yes"])
    st.markdown("<p class='field-desc'>Do you have diabetes?</p>", unsafe_allow_html=True)

# ==========================
# LIFESTYLE & DEMOGRAPHICS
# ==========================
with st.expander('Lifestyle & Demographics', expanded=True):
    # Collect user inputs for lifestyle and demographic details
    marital_status = st.selectbox("Marital Status", ["Married", "Single"])
    residence_type = st.selectbox("Residence Type", ["Rural", "Urban"])
    st.markdown("<p class='field-desc'>Do you live in a rural or urban area?</p>", unsafe_allow_html=True)

    work_type = st.selectbox(
        "Work Type",
        ["Self-employed", "Unemployed", "Private", "Government Job"]
    )
    smoking_status = st.selectbox(
        "Smoking Status",
        ["Non-smoker", "Formerly Smoker or Currently Smokes"]
    )

# ==========================
# VALIDATION & PREDICT BUTTON
# ==========================
with st.container():
    error_msgs = []
    
    # Validation checks for user inputs
    if not (100 <= height <= 250):
        error_msgs.append("• Height must be between 100 and 250 cm.")
    if not (30 <= weight <= 200):
        error_msgs.append("• Weight must be between 30 and 200 kg.")
    if not (12 <= bmi <= 60):  # BMI between 12 and 60
        error_msgs.append("• BMI value is out of a reasonable range. Please check your height and weight.")
    if not (18 <= age <= 100):  # Age between 18 and 100
        error_msgs.append("• Age must be between 18 and 100.")

    # Display validation errors or success message
    if error_msgs:
        st.error(
            "Please correct the following before predicting:\n\n" +
            "\n".join(error_msgs)
        )
        st.info("Once all fields are valid, you can click Predict.")
    else:
        st.success("All inputs look good! Click Predict to see your results.")

        # Use st.dialog for confirmation modal
        if st.button("Predict", disabled=bool(error_msgs)):
            # Convert variables to numeric values for model input
            hypertension_val = 1 if hypertension == "Yes" else 0
            heart_disease_val = 1 if heart_disease == "Yes" else 0
            diabetes_val = 1 if diabetes == "Yes" else 0
            marital_status_val = 1 if marital_status == "Married" else 0
            smoking_status_val = 1 if smoking_status == "Formerly Smoker or Currently Smokes" else 0
            
            @st.dialog("Please Confirm Your Details", width="large")
            def confirm_dialog():
                """
                Display a confirmation dialog for users to review their inputs.
                """
                st.markdown("#### Please review your information before submitting:")
                
                # --- Card 1: Personal Information ---
                st.markdown(
                    f"""
                    <div style='background-color:#f7fcfa;border-radius:10px;padding:1em;margin-bottom:1em;border-left:4px solid #2ec4b6;box-shadow:0 2px 6px rgba(0,0,0,0.1);'>
                    <h4 style='margin:0 0 0.5em 0;color:#1a3c40;font-size:1.1em;'>👤 Personal Information</h4>
                    <div style='display:flex;flex-wrap:wrap;gap:0.8em;font-size:0.95em;'>
                        <span style='background:#e8f5f2;padding:0.3em 0.6em;border-radius:15px;'><b>Age:</b> {age}</span>
                        <span style='background:#e8f5f2;padding:0.3em 0.6em;border-radius:15px;'><b>Gender:</b> {gender}</span>
                        <span style='background:#e8f5f2;padding:0.3em 0.6em;border-radius:15px;'><b>Height:</b> {height}cm</span>
                        <span style='background:#e8f5f2;padding:0.3em 0.6em;border-radius:15px;'><b>Weight:</b> {weight}kg</span>
                        <span style='background:#e8f5f2;padding:0.3em 0.6em;border-radius:15px;'><b>BMI:</b> {bmi:.1f}</span>
                    </div>
                    </div>
                    """, unsafe_allow_html=True
                )

                # --- Card 2: Medical History ---
                st.markdown(
                    f"""
                    <div style='background-color:#f7faff;border-radius:10px;padding:1em;margin-bottom:1em;border-left:4px solid #4361ee;box-shadow:0 2px 6px rgba(0,0,0,0.1);'>
                    <h4 style='margin:0 0 0.5em 0;color:#1a3c40;font-size:1.1em;'>🏥 Medical History</h4>
                    <div style='display:flex;flex-wrap:wrap;gap:0.8em;font-size:0.95em;'>
                        <span style='background:#e8f0ff;padding:0.3em 0.6em;border-radius:15px;'><b>Blood Pressure:</b> {hypertension}</span>
                        <span style='background:#e8f0ff;padding:0.3em 0.6em;border-radius:15px;'><b>Heart Disease:</b> {heart_disease}</span>
                        <span style='background:#e8f0ff;padding:0.3em 0.6em;border-radius:15px;'><b>Diabetes:</b> {diabetes}</span>
                    </div>
                    </div>
                    """, unsafe_allow_html=True
                )

                # --- Card 3: Lifestyle & Demographics ---
                st.markdown(
                    f"""
                    <div style='background-color:#fff9e6;border-radius:10px;padding:1em;margin-bottom:1.5em;border-left:4px solid #ffbe0b;box-shadow:0 2px 6px rgba(0,0,0,0.1);'>
                    <h4 style='margin:0 0 0.5em 0;color:#1a3c40;font-size:1.1em;'>🏠 Lifestyle & Demographics</h4>
                    <div style='display:flex;flex-wrap:wrap;gap:0.8em;font-size:0.95em;'>
                        <span style='background:#fff5cc;padding:0.3em 0.6em;border-radius:15px;'><b>Married:</b> {marital_status}</span>
                        <span style='background:#fff5cc;padding:0.3em 0.6em;border-radius:15px;'><b>Area:</b> {residence_type}</span>
                        <span style='background:#fff5cc;padding:0.3em 0.6em;border-radius:15px;'><b>Work:</b> {work_type}</span>
                        <span style='background:#fff5cc;padding:0.3em 0.6em;border-radius:15px;'><b>Smoking:</b> {smoking_status}</span>
                    </div>
                    </div>
                    """, unsafe_allow_html=True
                )

                # Buttons for editing or confirming inputs
                colA, colB = st.columns(2)
                with colA:
                    if st.button("Continue Editing"):
                        st.rerun()
                with colB:
                    if st.button("Confirm & Predict"):
                        # Process user inputs and encode them for the model
                        age_group = age_to_age_group(age)
                        age_gender_risk = age_gender_to_risk(age_group, gender)
                        health_risk = health_risk_level(hypertension_val, heart_disease_val, diabetes_val)
                        
                        work_type_mapped = "Employed" if work_type == "Government Job" else work_type
                        marital_status_text = "Yes" if marital_status == "Married" else "No"
                        stress_level = stress_level_category(work_type_mapped, marital_status_text, residence_type, health_risk)

                        # Create feature encoding for the model
                        user_inputs = {}
                        user_inputs["hypertension"] = hypertension_val
                        user_inputs["heart_disease"] = heart_disease_val  
                        user_inputs["ever_married"] = marital_status_val
                        user_inputs["smoking_status"] = smoking_status_val
                        user_inputs["diabetes"] = diabetes_val
                        
                        # Encode categorical features
                        age_groups = ["Middle (50-64)", "Older (65+)", "Young (<49)"]
                        for ag in age_groups:
                            user_inputs[f"age_group_{ag}"] = 1 if age_to_age_group(age) == ag else 0
                        
                        work_types = ["Employed", "Private", "Self-employed", "Unemployed"] 
                        for wt in work_types:
                            user_inputs[f"work_type_{wt}"] = 1 if work_type_mapped == wt else 0
                        
                        bmi_categories = ["Normal weight", "Obese"]
                        for bc in bmi_categories:
                            user_inputs[f"bmi_category_{bc}"] = 1 if bmi_category == bc else 0
                        
                        health_risks = ["Low Risk", "Moderate Risk"]
                        for hr in health_risks:
                            user_inputs[f"health_risk_{hr}"] = 1 if health_risk == hr else 0
                        
                        age_gender_risks = ["High Risk", "Low Risk", "Moderate Risk", "Very High Risk"]
                        for agr in age_gender_risks:
                            user_inputs[f"age_gender_risk_{agr}"] = 1 if age_gender_risk == agr else 0
                        
                        stress_levels = ["Low Stress", "Moderate Stress"]
                        for sl in stress_levels:
                            user_inputs[f"stress_level_{sl}"] = 1 if stress_level == sl else 0
                        
                        # Save inputs to session state and redirect to results page
                        st.session_state.user_inputs = user_inputs
                        st.success("Inputs confirmed! Redirecting to results...")
                        st.switch_page("pages/Results.py")
            confirm_dialog()

# ==========================
# FOOTER
# ==========================
footer()
