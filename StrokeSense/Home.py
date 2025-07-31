import streamlit as st
from config.theme import theme, app_background
from config.design import disclaimer, hero_section, how_to_use_section, load_lottie_file
from streamlit_lottie import st_lottie

# ==========================
# PAGE CONFIGURATION
# ==========================
st.set_page_config(
    page_title="StrokeSense",  # Title of the app
    page_icon="../assets/icon.png",  # Icon for the app
    layout="wide",  # Wide layout for better user experience
    initial_sidebar_state="expanded",  # Sidebar is expanded by default
)

# ==========================
# APPLY THEME AND BACKGROUND
# ==========================
app_background()  # Apply background animation
theme()  # Apply the app's theme

# ==========================
# HERO SECTION
# ==========================
hero_section()  # Display the hero section at the top of the page

# Load the Lottie animation for the shield icon
lottie_shield = load_lottie_file("../assets/shield.json")

# Hero section with a styled container
with st.container():
    col1, col2 = st.columns([1, 1])  # Split the container into two equal columns
    with col1:
        # Display the main title and description
        st.markdown("""
            <div class="hero-container">
                <h1>Stroke Risk Prediction</h1>
                <p>Welcome to StrokeSense, your personal stroke risk prediction app.</p>
                <p>Understand your risk. Get actionable insights. Take control of your health.</p>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        # Display the Lottie animation in the second column
        st_lottie(lottie_shield, height=350, key="shield_animation", 
                  speed=1, width=400, quality="high", loop=True)

# ==========================
# GET STARTED BUTTON
# ==========================
# Button to navigate to the input form page
if st.button("Get Started"):
    st.switch_page("pages/Input.py")  # Redirect to the input form page

# ==========================
# MEDICAL DISCLAIMER
# ==========================
disclaimer()  # Display a medical disclaimer for the app

# ==========================
# HOW TO USE SECTION
# ==========================
how_to_use_section()  # Display instructions on how to use the app

# ==========================
# FEATURES SECTION
# ==========================
# Highlight the app's features in a styled section
st.markdown("""
    <div class="features">
        <h3>Why Use This App?</h3>
        <div class="feature-card">
            <span class="material-icons">insights</span>
            <h4>Smart Predictions</h4>
            <p>Leverages advanced machine learning to deliver accurate stroke risk estimation tailored to you.</p>
        </div>
        <div class="feature-card">
            <span class="material-icons">favorite</span>
            <h4>Personalised Advice</h4>
            <p>Receive health tips based on your unique profile from managing BMI to improving lifestyle habits.</p>
        </div>
        <div class="feature-card">
            <span class="material-icons">health_and_safety</span>
            <h4>Prevention First</h4>
            <p>Empowers you with actionable recommendations to lower your risk and stay ahead of potential health concerns.</p>
        </div>
        <div class="feature-card">
            <span class="material-icons">query_stats</span>
            <h4>Visual Risk Insights</h4>
            <p>Interactive charts and gauges help you easily understand your risk levels and progress over time.</p>
        </div>
        <div class="feature-card">
            <span class="material-icons">lightbulb</span>
            <h4>Explainable Results</h4>
            <p>Breaks down which factors contribute most to your risk so you can take targeted action.</p>
        </div>
    </div>
""", unsafe_allow_html=True)

# ==========================
# FOOTER
# ==========================
# Add a footer with a custom style
st.markdown("""
    <footer class="custom-footer">
        <p>Made with ❤️ by <b>StrokeSense Team</b></p>
    </footer>
    <style>
        .custom-footer {
            text-align: center;
            padding: 25px 10px;
            color: #222;
            font-size: 1.1rem;
            border-top: 2px solid #d0d0d0;
            border-radius: 0 0 16px 16px;
            box-shadow: 0 -2px 10px rgba(0,0,0,0.05);
            margin-top: 50px;
        }
        .custom-footer p {
            margin: 0;
            font-weight: 500;
        }
        .custom-footer b {
            color: #00796B; /* Accent color for the team name */
        }
    </style>
""", unsafe_allow_html=True)



