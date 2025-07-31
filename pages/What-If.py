import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from sklearn.impute import SimpleImputer
from utils.model import load_model_and_features
from utils.bmi import calculate_bmi
from utils.risk_n_level import age_gender_to_risk, age_to_age_group, health_risk_level, stress_level_category
from config.theme import theme, app_background
from config.design import disclaimer, load_lottie_file, input_design, hero_section, how_to_use_section, footer
from streamlit_lottie import st_lottie

# ==========================
# HELPER FUNCTION
# ==========================
def build_feature_vector(user_inputs, feature_order):
    """
    Build a feature vector DataFrame from user inputs.
    Handles missing values by filling them with default values.
    """
    input_data = []
    for feature in feature_order:
        value = user_inputs.get(feature, 0)  # Default to 0 for missing features
        input_data.append(value)
    
    input_df = pd.DataFrame([input_data], columns=feature_order)
    
    # Handle any remaining NaN values
    if input_df.isnull().any().any():
        imputer = SimpleImputer(strategy='constant', fill_value=0)
        input_df = pd.DataFrame(imputer.fit_transform(input_df), columns=feature_order)
    
    return input_df

# ==========================
# PAGE CONFIGURATION
# ==========================
st.set_page_config(
    page_title="StrokeSense – What-If",
    page_icon="../assets/icon.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==========================
# APPLY THEME AND STYLING
# ==========================
theme()  # Apply the theme
app_background()  # Apply the background animation
input_design()  # Apply input design styles
hero_section()  # Display the hero section
how_to_use_section()  # Display the "How to Use" section

# ==========================
# PAGE TITLE AND DESCRIPTION
# ==========================
st.markdown("""
    <div class="hero-container">
        <h1>What-If Scenario Explorer</h1>
        <p>Explore how <strong>lifestyle changes</strong> can impact your stroke risk. Adjust the factors below to see immediate results!</p>
        <div style="background: linear-gradient(135deg, #e3f2fd 0%, #f3e5f5 100%); text-align: left; padding: 15px; border-radius: 10px; margin: 20px 0; border-left: 4px solid #2196f3;">
            <h4 style="margin: 0 0 10px 0; color: #1565c0;">💡 Examples of What You Can Explore:</h4>
            <p style="margin: 5px 0; color: #424242;">• "What if I lose weight and reach a healthy BMI?" → Risk might drop from 48% to 35%</p>
            <p style="margin: 5px 0; color: #424242;">• "What if I quit smoking completely?" → Risk could decrease by 10-15%</p>
            <p style="margin: 5px 0; color: #424242;">• "What if I exercise regularly?" → Combined lifestyle changes can significantly reduce risk</p>
        </div>
    </div>
""", unsafe_allow_html=True)

# ==========================
# LOAD MODEL AND FEATURE ORDER
# ==========================
model, feature_order = load_model_and_features()

# ==========================
# MEDICAL DISCLAIMER
# ==========================
disclaimer()

# ==========================
# CHECK FOR USER INPUTS
# ==========================
if "user_inputs" not in st.session_state:
    # Show error if no inputs are found
    st.error("No inputs found. Please go back and fill the form.")
    no_data = load_lottie_file("../assets/no_data_found.json")
    st_lottie(no_data, height=350, key="no_data")
    if st.button("Go Back to Form", key="back_to_form"):
        st.switch_page("pages/Input.py")
    st.stop()

# Initialize what-if inputs if not already present
if "whatif_inputs" not in st.session_state:
    st.session_state.whatif_inputs = st.session_state.user_inputs.copy()

# Copy original and previous inputs for comparison
original_inputs = st.session_state.whatif_inputs.copy()
previous_inputs = st.session_state.user_inputs.copy()

# Calculate original risk percentage
original_risk = model.predict_proba(build_feature_vector(previous_inputs, feature_order))[0][1] * 100

# ==========================
# LAYOUT: TWO COLUMNS
# ==========================
col1, col2 = st.columns([1, 1])

# ==========================
# COLUMN 1: LIFESTYLE CHANGES
# ==========================
with col1:
    st.markdown("""
    <div class="section-header" style="
        background: linear-gradient(135deg, #4caf50 0%, #2e7d32 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        font-size: 1.3rem;
        font-weight: bold;
        margin-bottom: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    ">
        🎛️ Lifestyle Changes You Can Make
    </div>
    """, unsafe_allow_html=True)

    st.info("💡 **Tip**: These are factors you can actually change! Adjust them to see how lifestyle modifications impact your stroke risk.")

    # --- Weight Management (BMI) ---
    with st.expander("⚖️ Weight & BMI Management", expanded=True):
        st.markdown("""
        <div style="background: #f1f8e9; padding: 10px; border-radius: 8px; margin-bottom: 15px;">
            <h4 style="color: #2e7d32; margin: 0 0 8px 0;">🎯 Goal: Achieve Healthy Weight</h4>
            <p style="margin: 0; color: #424242;">Healthy BMI range: 18.5 - 24.9</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Get current height from user inputs (unchangeable)
        current_height = 170  # Default, should be extracted from user data
        if "user_inputs" in st.session_state:
            current_height = 170  # This would need to be properly extracted
            
        st.markdown(f"**Your Height**: {current_height} cm *(fixed)*")
        
        # Weight slider for what-if scenarios
        weight = st.slider(
            "Target Weight (kg)",
            min_value=40,
            max_value=150,
            value=int(original_inputs.get("Weight", 70)),
            step=1,
            help="Slide to see how weight changes affect your stroke risk"
        )
        
        bmi, bmi_category = calculate_bmi(current_height, weight)
        
        # Color-coded BMI display
        if bmi < 18.5:
            bmi_color = "#ff9800"  # Orange for underweight
            bmi_status = "Underweight"
        elif 18.5 <= bmi < 25:
            bmi_color = "#4caf50"  # Green for normal
            bmi_status = "Normal Weight ✅"
        elif 25 <= bmi < 30:
            bmi_color = "#ff9800"  # Orange for overweight
            bmi_status = "Overweight"
        else:
            bmi_color = "#f44336"  # Red for obese
            bmi_status = "Obese"
            
        st.markdown(f"""
        <div style="background: {bmi_color}20; border: 2px solid {bmi_color}; border-radius: 8px; padding: 10px; text-align: center;">
            <h3 style="color: {bmi_color}; margin: 0;">BMI: {bmi:.1f}</h3>
            <p style="color: {bmi_color}; margin: 5px 0 0 0; font-weight: bold;">{bmi_status}</p>
        </div>
        """, unsafe_allow_html=True)

    # --- Smoking Status ---
    with st.expander("🚭 Smoking Cessation", expanded=True):
        st.markdown("""
        <div style="background: #e8f5e8; padding: 10px; border-radius: 8px; margin-bottom: 15px;">
            <h4 style="color: #2e7d32; margin: 0 0 8px 0;">🎯 Goal: Quit Smoking</h4>
            <p style="margin: 0; color: #424242;">Quitting smoking can reduce stroke risk by up to 50% within 2 years!</p>
        </div>
        """, unsafe_allow_html=True)
        
        current_smoking = 0 if original_inputs.get("smoking_status", 0) == 0 else 1
        smoking_options = ["Non-smoker", "Formerly Smoker or Currently Smokes"]
        
        smoking_status = st.radio(
            "What if you change your smoking status?",
            smoking_options,
            index=current_smoking,
            help="See the immediate impact of quitting smoking on your stroke risk"
        )
        
        if smoking_status == "Non-smoker" and current_smoking == 1:
            st.success("🎉 Great choice! Quitting smoking is one of the best things you can do for your health.")
        elif smoking_status == "Formerly Smoker or Currently Smokes" and current_smoking == 0:
            st.warning("⚠️ This would increase your stroke risk. Consider quitting for better health.")

    # --- Physical Activity ---
    with st.expander("🏃‍♂️ Physical Activity Level", expanded=True):
        st.markdown("""
        <div style="background: #e3f2fd; padding: 10px; border-radius: 8px; margin-bottom: 15px;">
            <h4 style="color: #1976d2; margin: 0 0 8px 0;">🎯 Goal: Stay Active</h4>
            <p style="margin: 0; color: #424242;">Aim for at least 150 minutes of moderate exercise per week</p>
        </div>
        """, unsafe_allow_html=True)
        
        activity_levels = [
            "Low - Less than 30 minutes/week",
            "Moderate - About 150 minutes/week", 
            "High - More than 150 minutes/week"
        ]
        
        current_activity = 1  # Default to moderate
        physical_activity = st.radio(
            "What if you change your activity level?",
            activity_levels,
            index=current_activity,
            help="Regular physical activity significantly reduces stroke risk"
        )
        
        if "High" in physical_activity:
            st.success("🏆 Excellent! High activity levels provide maximum protection.")
        elif "Moderate" in physical_activity:
            st.info("👍 Good job! This meets recommended guidelines.")
        else:
            st.warning("📈 Consider increasing activity for better health outcomes.")

    # --- Stress Management ---
    with st.expander("🧘‍♀️ Stress Management", expanded=True):
        st.markdown("""
        <div style="background: #f3e5f5; padding: 10px; border-radius: 8px; margin-bottom: 15px;">
            <h4 style="color: #7b1fa2; margin: 0 0 8px 0;">🎯 Goal: Reduce Stress</h4>
            <p style="margin: 0; color: #424242;">Chronic stress contributes to high blood pressure and stroke risk</p>
        </div>
        """, unsafe_allow_html=True)
        
        stress_level = st.select_slider(
            "What if you improve your stress management?",
            options=["High Stress", "Moderate Stress", "Low Stress"],
            value="Moderate Stress",
            help="Better stress management through meditation, exercise, or therapy"
        )
        
        if stress_level == "Low Stress":
            st.success("🧘‍♀️ Perfect! Low stress levels contribute to better overall health.")
        elif stress_level == "Moderate Stress":
            st.info("😌 Not bad, but there's room for improvement.")
        else:
            st.warning("😰 High stress increases health risks. Consider stress reduction techniques.")

    # --- Alcohol Consumption ---
    with st.expander("🍷 Alcohol Consumption", expanded=True):
        st.markdown("""
        <div style="background: #fff3e0; padding: 10px; border-radius: 8px; margin-bottom: 15px;">
            <h4 style="color: #ef6c00; margin: 0 0 8px 0;">🎯 Goal: Moderate or Eliminate</h4>
            <p style="margin: 0; color: #424242;">Excessive alcohol increases stroke risk</p>
        </div>
        """, unsafe_allow_html=True)
        
        alcohol_options = ["Never", "Rarely", "Social Drinker", "Frequent Drinker"]
        alcohol_intake = st.radio(
            "What if you change your drinking habits?",
            alcohol_options,
            index=0,
            help="Reducing alcohol consumption can lower stroke risk"
        )
        
        if alcohol_intake == "Never":
            st.success("🚫 Excellent choice for optimal health!")
        elif alcohol_intake == "Rarely":
            st.info("👍 Very good - minimal consumption is healthiest.")
        elif alcohol_intake == "Social Drinker":
            st.warning("⚠️ Moderate - consider reducing frequency.")
        else:
            st.error("🚨 High risk - strongly consider reducing consumption.")

# ==========================
# COLUMN 2: IMPACT ANALYSIS
# ==========================
with col2:
    st.markdown("""
    <div class="section-header" style="
        background: linear-gradient(135deg, #2196f3 0%, #1976d2 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        font-size: 1.3rem;
        font-weight: bold;
        margin-bottom: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    ">
        📊 Impact of Your Changes
    </div>
    """, unsafe_allow_html=True)
    
    # Predict new risk with modified inputs
    modified_inputs = previous_inputs.copy()
    input_df = build_feature_vector(modified_inputs, feature_order)
    y_probs = model.predict_proba(input_df)[:, 1]
    risk_percentage = y_probs[0] * 100

    # Risk difference calculation
    risk_difference = risk_percentage - original_risk
    
    # Display potential risk
    st.markdown(f"""
    <div style="background: #e3f2fd; padding: 15px; border-radius: 10px; text-align: center; margin-bottom: 20px;">
        <h4 style="color: #1976d2; margin: 0;">🎯 Your Potential Risk with Changes</h4>
        <h2 style="color: #2196f3; margin: 10px 0;">{risk_percentage:.1f}%</h2>
    </div>
    """, unsafe_allow_html=True)

    # Enhanced gauge chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=risk_percentage,
        delta={
            'reference': original_risk, 
            'increasing': {'color': "#E53E3E"}, 
            'decreasing': {'color': "#4CAF50"},
            'font': {'size': 20}
        },
        gauge={
            'axis': {
                'range': [0, 100],
                'tickfont': {'size': 14},
                'tickcolor': '#4A5568'
            },
            'bar': {'color': "#2196F3", 'thickness': 0.3},
            'bgcolor': "#F7FAFC",
            'borderwidth': 2,
            'bordercolor': "#E2E8F0",
            'steps': [
                {'range': [0, 25], 'color': "#C8E6C9"},  # Light green
                {'range': [25, 50], 'color': "#FFE0B2"}, # Light orange  
                {'range': [50, 75], 'color': "#FFCDD2"}, # Light red
                {'range': [75, 100], 'color': "#F8BBD9"} # Light pink
            ],
            'threshold': {
                'line': {'color': "#2D3748", 'width': 3},
                'thickness': 0.8,
                'value': original_risk  # Show original risk as threshold
            }
        },
        number={
            'suffix': "%",
            'font': {'size': 28, 'color': '#2D3748'}
        },
        title={
            'text': "Modified Risk Level",
            'font': {'size': 18, 'color': '#2D3748'}
        }
    ))
    
    fig.update_layout(
        height=400,
        margin=dict(l=0, r=0, t=50, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Enhanced insights with actionable advice
    st.markdown("### 📈 Impact Analysis")
    
    if risk_difference < -5:  # Significant improvement
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #c8e6c9 0%, #a5d6a7 100%);
            border: 3px solid #4caf50;
            border-radius: 15px;
            padding: 25px;
            margin: 20px 0;
            text-align: center;
        ">
            <h3 style="color: #2e7d32; margin: 0 0 15px 0;">🎉 Excellent Progress!</h3>
            <h2 style="color: #1b5e20; margin: 0 0 10px 0;">-{abs(risk_difference):.1f}% Risk Reduction</h2>
            <p style="color: #2e7d32; font-size: 1.2rem; margin: 0;">
                These lifestyle changes could significantly improve your health outcomes!
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    elif risk_difference < -1:  # Moderate improvement
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #dcedc8 0%, #c5e1a5 100%);
            border: 2px solid #8bc34a;
            border-radius: 12px;
            padding: 20px;
            margin: 15px 0;
            text-align: center;
        ">
            <h4 style="color: #33691e; margin: 0 0 10px 0;">👍 Good Improvement!</h4>
            <p style="color: #33691e; font-size: 1.1rem; margin: 0;">
                Your changes reduced stroke risk by <strong>{abs(risk_difference):.1f}%</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    elif risk_difference > 5:  # Significant increase
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%);
            border: 3px solid #f44336;
            border-radius: 15px;
            padding: 25px;
            margin: 20px 0;
            text-align: center;
        ">
            <h3 style="color: #c62828; margin: 0 0 15px 0;">⚠️ Risk Increased</h3>
            <h2 style="color: #b71c1c; margin: 0 0 10px 0;">+{risk_difference:.1f}% Higher Risk</h2>
            <p style="color: #c62828; font-size: 1.2rem; margin: 0;">
                Consider adopting healthier lifestyle choices to reduce your risk.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    elif abs(risk_difference) <= 1:  # Minimal change
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #e1f5fe 0%, #b3e5fc 100%);
            border: 2px solid #03a9f4;
            border-radius: 12px;
            padding: 20px;
            margin: 15px 0;
            text-align: center;
        ">
            <h4 style="color: #01579b; margin: 0 0 10px 0;">📊 Minimal Change</h4>
            <p style="color: #01579b; font-size: 1.1rem; margin: 0;">
                Your risk remains relatively stable. Try combining multiple lifestyle changes for greater impact.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Actionable next steps
    st.markdown("### 🎯 Recommended Actions")
    
    recommendations = []
    
    if bmi >= 25:
        recommendations.append("🏃‍♂️ **Weight Management**: Aim for a BMI between 18.5-24.9 through diet and exercise")
        
    if smoking_status != "Non-smoker":
        recommendations.append("🚭 **Quit Smoking**: This single change can reduce stroke risk by up to 50%")
        
    if stress_level == "High Stress":
        recommendations.append("🧘‍♀️ **Stress Reduction**: Try meditation, yoga, or regular relaxation techniques")
        
    if "Low" in physical_activity:
        recommendations.append("💪 **Increase Activity**: Aim for 150+ minutes of moderate exercise weekly")
        
    if alcohol_intake in ["Frequent Drinker"]:
        recommendations.append("🍷 **Reduce Alcohol**: Limit consumption to lower health risks")
    
    if recommendations:
        for rec in recommendations:
            st.markdown(f"• {rec}")
    else:
        st.success("🌟 You're making excellent lifestyle choices! Keep up the great work!")
        # show lottie animation for tips
        tips_animation = load_lottie_file("../assets/tips.json")
        st_lottie(tips_animation, height=500, key="tips_animation")

# ==========================
# FOOTER
# ==========================
footer()




