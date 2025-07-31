import numpy as np
import streamlit as st
import pandas as pd
from config.theme import theme, app_background
from config.design import disclaimer, how_to_use_section, load_lottie_file, footer
from utils.model import load_model_and_features
from sklearn.impute import SimpleImputer

import plotly.express as px
import plotly.graph_objects as go
from streamlit_lottie import st_lottie
import time

# ==========================
# PAGE CONFIGURATION
# ==========================
st.set_page_config(
    page_title="StrokeSense – Results",
    page_icon="../assets/icon.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==========================
# APPLY THEME AND STYLING
# ==========================
theme()  # Apply the theme
app_background()  # Apply the background animation
how_to_use_section()  # Display the "How to Use" section
st.title("Stroke Risk Prediction – Results")

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

# ==========================
# LOAD MODEL AND PREDICT
# ==========================
model, feature_order = load_model_and_features()
with st.spinner("Generating your stroke risk result..."):
    time.sleep(2)  # Simulate a delay for better user experience
    
    # Prepare input data for the model
    input_data = []
    for feature in feature_order:
        value = st.session_state.user_inputs.get(feature, 0)  # Default to 0 for missing features
        input_data.append(value)
    
    input_df = pd.DataFrame([input_data], columns=feature_order)
    
    # Handle missing values in the input data
    if input_df.isnull().any().any():
        imputer = SimpleImputer(strategy='constant', fill_value=0)
        input_df = pd.DataFrame(imputer.fit_transform(input_df), columns=feature_order)
    
    # Predict probabilities
    y_probs = model.predict_proba(input_df)[:, 1]

    # Apply custom threshold for classification
    custom_threshold = 0.55
    y_pred = (y_probs > custom_threshold).astype(int)  # Risk is 1 if probability > threshold
    risk_percentage = y_probs[0] * 100  # Convert probability to percentage for display

# ==========================
# DISPLAY RISK LEVEL
# ==========================
# Determine risk level and display appropriate message
if risk_percentage < 20:
    color = "#28a745"   # Green for low risk
    st.success("Low risk – Keep up the healthy habits!")
elif risk_percentage < 50:
    color = "#ffc107"   # Yellow for moderate risk
    st.warning("Moderate risk – Consider lifestyle changes & regular check-ups.")
else:
    color = "#dc3545"   # Red for high risk
    st.error("High risk – Please consult a healthcare provider soon.")

# Display the risk percentage in a styled card
st.markdown(f"""
    <div style="
        background-color: {color};
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        color: white;
        font-size: 2rem;
        font-weight: bold;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        margin-bottom: 20px;
    ">
        Your Estimated Stroke Risk: {risk_percentage:.1f}%
    </div>
""", unsafe_allow_html=True)

# Display medical disclaimer
disclaimer()

# ==========================
# RISK LADDER VISUALIZATION
# ==========================
# Create a visual representation of the user's risk on a ladder
categories = ["Low", "Average", "High", "Critical"]
ranges = [0, 20, 50, 75, 100]
colors = ["#28a745", "#ffc107", "#fd7e14", "#dc3545"]

fig = go.Figure()

# Add ladder segments
for i in range(len(categories)):
    fig.add_trace(go.Bar(
        x=[ranges[i+1] - ranges[i]],
        y=["Risk Ladder"],
        orientation='h',
        marker=dict(color=colors[i]),
        name=f"{categories[i]} ({ranges[i]}–{ranges[i+1]}%)",
        hovertemplate=f"{categories[i]} Risk: {ranges[i]}–{ranges[i+1]}%"
    ))

# Add user's risk as a vertical line
fig.add_shape(
    type="line",
    x0=risk_percentage, x1=risk_percentage,
    y0=-0.5, y1=0.5,
    line=dict(color="black", width=4, dash="dash"),
)
fig.add_annotation(
    x=risk_percentage,
    y=0.2,
    text=f"Your Risk: {risk_percentage:.1f}%",
    showarrow=False,
    font=dict(color="black", size=14, family="Arial"),
    bgcolor="white"
)

# Configure the layout of the chart
fig.update_layout(
    barmode='stack',
    height=200,
    title="Stroke Risk Ladder",
    xaxis=dict(title="Stroke Risk (%)", range=[0, 100], showgrid=False),
    yaxis=dict(showticklabels=False),
    plot_bgcolor="white",
    showlegend=True,
    margin=dict(l=40, r=40, t=60, b=40)
)

st.plotly_chart(fig, use_container_width=True)

# ==========================
# PERSONALIZED INSIGHTS
# ==========================
# Identify active risk factors based on user inputs
user_values = [st.session_state.user_inputs.get(feat, 0) for feat in feature_order]
active_risk_factors = [feat for feat, val in zip(feature_order, user_values) if val == 1]

# Map feature names to user-friendly names
feature_name_mapping = {
    "hypertension": "High Blood Pressure",
    "heart_disease": "Heart Disease", 
    "ever_married": "Married",
    "smoking_status": "Smoking History",
    "diabetes": "Diabetes",
    "bmi_category_Obese": "BMI: Obese",
    "bmi_category_Normal weight": "BMI: Normal",
    "stress_level_Moderate Stress": "Moderate Stress Level",
    "stress_level_Low Stress": "Low Stress Level",
    "age_group_Young (<49)": "Age: Under 49",
    "age_group_Middle (50-64)": "Age: 50-64", 
    "age_group_Older (65+)": "Age: 65+",
    "health_risk_Low Risk": "Low Health Risk",
    "health_risk_Moderate Risk": "Moderate Health Risk",
    "age_gender_risk_Low Risk": "Low Age-Gender Risk",
    "age_gender_risk_Moderate Risk": "Moderate Age-Gender Risk", 
    "age_gender_risk_High Risk": "High Age-Gender Risk",
    "age_gender_risk_Very High Risk": "Very High Age-Gender Risk",
    "work_type_Private": "Private Sector Work",
    "work_type_Employed": "Government Work",
    "work_type_Self-employed": "Self-Employed", 
    "work_type_Unemployed": "Unemployed"
}

# ==========================
# EDUCATIONAL TABS
# ==========================
# Provide educational content in tabs
st.markdown("---")
st.markdown("## 🧠 Stroke Risk Education")

tab1, tab2, tab3, tab4 = st.tabs(["📊 Risk Statistics", "🏥 Prevention Tips", "⚠️ Warning Signs", "📚 Learn More"])

# Tab 1: Risk Statistics
with tab1:
    st.markdown("### Global Stroke Statistics")
    # Display stroke statistics using bar charts
    stroke_stats = {
        'Age Group': ['18-44', '45-64', '65-74', '75+'],
        'Stroke Risk (%)': [2, 8, 15, 25],
        'Prevention Potential (%)': [80, 70, 60, 40]
    }
    stats_df = pd.DataFrame(stroke_stats)
    col1, col2 = st.columns(2)
    with col1:
        fig_risk = px.bar(stats_df, x='Age Group', y='Stroke Risk (%)', 
                         title='Stroke Risk by Age Group',
                         color='Stroke Risk (%)',
                         color_continuous_scale='Reds')
        fig_risk.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_risk, use_container_width=True)
    with col2:
        fig_prevention = px.bar(stats_df, x='Age Group', y='Prevention Potential (%)', 
                               title='Prevention Potential by Age',
                               color='Prevention Potential (%)',
                               color_continuous_scale='Greens')
        fig_prevention.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_prevention, use_container_width=True)
    st.info("💡 **Key Insight**: Up to 80% of strokes are preventable through lifestyle changes!")

# Tab 2: Prevention Tips
with tab2:
    st.markdown("### 🎯 Personalized Prevention Tips")
    # Provide personalized prevention tips based on user inputs
    prevention_tips = []
    if st.session_state.user_inputs.get("hypertension", 0) == 1:
        prevention_tips.append({
            'icon': '🩺',
            'title': 'Blood Pressure Management',
            'tip': 'Monitor your blood pressure regularly and take prescribed medications consistently.',
            'action': 'Aim for <120/80 mmHg'
        })
    if st.session_state.user_inputs.get("bmi_category_Obese", 0) == 1:
        prevention_tips.append({
            'icon': '🏃‍♂️',
            'title': 'Weight Management',
            'tip': 'Focus on gradual weight loss through diet and exercise.',
            'action': 'Aim to lose 1-2 pounds per week'
        })
    if st.session_state.user_inputs.get("smoking_status", 0) == 1:
        prevention_tips.append({
            'icon': '🚭',
            'title': 'Smoking Cessation',
            'tip': 'Quitting smoking can reduce stroke risk by 50% within 2 years.',
            'action': 'Contact a smoking cessation program'
        })
    if st.session_state.user_inputs.get("stress_level_Moderate Stress", 0) == 1:
        prevention_tips.append({
            'icon': '🧘‍♀️',
            'title': 'Stress Management',
            'tip': 'Practice relaxation techniques like meditation or deep breathing.',
            'action': '10 minutes daily meditation'
        })
    # Add general tips if no specific risk factors
    if not prevention_tips:
        prevention_tips = [
            {'icon': '🥗', 'title': 'Healthy Diet', 'tip': 'Eat more fruits, vegetables, and whole grains.', 'action': '5 servings of fruits/vegetables daily'},
            {'icon': '💊', 'title': 'Regular Checkups', 'tip': 'Monitor blood pressure, cholesterol, and blood sugar.', 'action': 'Annual health screening'},
            {'icon': '🏊‍♀️', 'title': 'Stay Active', 'tip': 'Regular exercise strengthens your cardiovascular system.', 'action': '150 minutes moderate exercise weekly'}
        ]
    for tip in prevention_tips:
        st.markdown(f"""
        <div style='background-color:#f0f8ff;border-radius:10px;padding:15px;margin:10px 0;border-left:5px solid #4169e1;'>
            <h4>{tip['icon']} {tip['title']}</h4>
            <p>{tip['tip']}</p>
            <strong>Action: {tip['action']}</strong>
        </div>
        """, unsafe_allow_html=True)

# Tab 3: Warning Signs
with tab3:
    st.markdown("### ⚠️ Stroke Warning Signs - F.A.S.T.")
    st.markdown("""
    <div style='background-color:#fff3cd;border:1px solid #ffeaa7;border-radius:10px;padding:20px;margin:10px 0;'>
        <h3 style='color:#856404;margin-top:0;'>Remember F.A.S.T.:</h3>
        <div style='display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:15px;margin-top:15px;'>
            <div style='text-align:center;'>
                <div style='font-size:3em;'>😵</div>
                <h4 style='color:#856404;'>F - Face</h4>
                <p>Face drooping or numbness</p>
            </div>
            <div style='text-align:center;'>
                <div style='font-size:3em;'>💪</div>
                <h4 style='color:#856404;'>A - Arms</h4>
                <p>Arm weakness or numbness</p>
            </div>
            <div style='text-align:center;'>
                <div style='font-size:3em;'>🗣️</div>
                <h4 style='color:#856404;'>S - Speech</h4>
                <p>Speech difficulty or slurred</p>
            </div>
            <div style='text-align:center;'>
                <div style='font-size:3em;'>⏰</div>
                <h4 style='color:#856404;'>T - Time</h4>
                <p>Time to call emergency services</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.error("🚨 **EMERGENCY**: If you notice these signs, call 911/emergency services immediately!")

# Tab 4: Learn More
with tab4:
    st.markdown("### 📚 Additional Resources (Singapore)")
    st.markdown("#### 🏥 Health Organizations")
    st.markdown("""
    <div class="features">
        <div class="feature-card">
            <span class="material-icons">local_hospital</span>
            <h4>Singapore Heart Foundation</h4>
            <p><a href="https://www.myheart.org.sg" target="_blank">myheart.org.sg</a></p>
        </div>
        <div class="feature-card">
            <span class="material-icons">health_and_safety</span>
            <h4>HealthHub (MOH)</h4>
            <p><a href="https://www.healthhub.sg" target="_blank">healthhub.sg</a></p>
        </div>
        <div class="feature-card">
            <span class="material-icons">coronavirus</span>
            <h4>National Neuroscience Institute</h4>
            <p><a href="https://www.nni.com.sg" target="_blank">nni.com.sg</a></p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("#### 📱 Mobile Apps for Health Tracking")
    st.markdown("""
    <div class="features">
        <div class="feature-card">
            <span class="material-icons">fitness_center</span>
            <h4>Healthy 365</h4>
            <p>Track steps, health points & nutrition (by HPB)</p>
        </div>
        <div class="feature-card">
            <span class="material-icons">self_improvement</span>
            <h4>Headspace</h4>
            <p>Meditation and stress management</p>
        </div>
        <div class="feature-card">
            <span class="material-icons">monitor_heart</span>
            <h4>Samsung Health / Apple Health</h4>
            <p>Monitor heart rate & physical activity</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("#### 🚨 Emergency Contacts")
    st.markdown("""
    <div class="features">
        <div class="feature-card">
            <span class="material-icons">emergency</span>
            <h4>Emergency Ambulance</h4>
            <p><strong>995</strong> (Singapore Civil Defence Force)</p>
        </div>
        <div class="feature-card">
            <span class="material-icons">support</span>
            <h4>HealthLine</h4>
            <p><strong>1800-223-1313</strong> (Non-emergency health advice)</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ==========================
# FOOTER AND NAVIGATION
# ==========================
# Add navigation buttons for "Go Back to Form" and "What If Analysis"
col1, col2 = st.columns([1, 1])
with col1:
    if st.button("Go Back to Form", key="back_to_form"):
        st.switch_page("pages/Input.py")  # Navigate back to the form page
with col2:
    if st.button("What If Analysis", key="what_if_analysis"):
        st.switch_page("pages/What-If.py")  # Navigate to the What-If page

# Footer
footer()


