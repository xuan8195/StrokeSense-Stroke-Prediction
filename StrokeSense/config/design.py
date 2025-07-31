import json
import streamlit as st
from streamlit_lottie import st_lottie

# Function to load Lottie animations from a local JSON file
def load_lottie_file(filepath: str):
    """Load a Lottie animation file from the given filepath."""
    with open(filepath, "r") as f:
        return json.load(f)

# Function to display the hero section at the top of the page
def hero_section():
    """Display the hero section with a title, description, and animation."""
    # Add custom styling for the hero section layout
    st.markdown("""
        <link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined" rel="stylesheet">
        <style> 
            .hero-container {
                background: linear-gradient(135deg, #E8F8FA, #DFF8EB);
                padding: 40px;
                border-radius: 20px;
                text-align: center;
                box-shadow: 0 4px 20px rgba(0,0,0,0.05);
                margin-bottom: 20px;
                width: 100%;
                margin-left: auto;
                margin-right: auto;
            }
            .hero-container h1 {
                font-size: 2.5rem;
                font-weight: 700;
                color: #222;
                margin-bottom: 10px;
                word-break: normal;
                white-space: normal;
                line-height: 1.2;
            }
            .hero-container p {
                font-size: 1.2rem;
                color: #555;
                margin-bottom: 0;
            }
            @media (max-width: 600px) {
                .hero-container {
                    padding: 18px;
                    border-radius: 12px;
                }
                .hero-container h1 {
                    font-size: 1.5rem;
                }
                .hero-container p {
                    font-size: 1rem;
                }
            }
        </style>
    """, unsafe_allow_html=True)

# Function to display the "How to Use" section with feature cards
def how_to_use_section():
    """Display the 'How to Use' section with feature cards."""
    st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined" rel="stylesheet">
    <style>
    /* Features Section */
    .features {
        margin-top: 3rem;
        text-align: center;
    }
    .feature-card {
        display: inline-block;
        background: #f9f9f9;
        padding: 20px;
        margin: 1rem;
        background: linear-gradient(135deg, #d4f1f4 0%, rgba(255, 255, 255, 0.9) 100%);
        border-radius: 20px;
        text-align: center;
        box-shadow:
            0 8px 24px rgba(0, 0, 0, 0.05),
            0 2px 8px rgba(40, 167, 69, 0.2);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        min-height: 350px;
        max-width: 300px;
        flex-shrink: 0;
        position: relative;
        overflow: hidden;
        -webkit-backdrop-filter: blur(10px);
        backdrop-filter: blur(10px);
    }
    .feature-card:hover {
        transform: translateY(-12px) scale(1.03);
        box-shadow:
            0 16px 48px rgba(0, 0, 0, 0.12),
            0 8px 20px rgba(40, 167, 69, 0.25);
        border: 1px solid #5298D2;
    }
    .material-icons {
        font-size: 48px;
        color: #28a745;
        margin-bottom: 10px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
        background: #e9f7ef;
        width: 90px;
        height: 90px;
        margin: 0 auto 1.5rem;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    </style>
    """, unsafe_allow_html=True)

# Function to style the input form with advanced design
def input_design():
    """Advanced design for input form using expander cards with Google Icons."""
    st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined" rel="stylesheet">
    <style>
        h1 {
            color: #28a745;
            text-align: center;
            font-weight: 700;
            margin-bottom: 2rem;
        }
        /* Expander as Card */
        div[data-testid="stExpander"] {
            background: #ffffff !important;
            border-radius: 14px !important;
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
            border: 1px solid #e5e5e5;
            margin-bottom: 1.5rem;
        }
        div[data-testid="stExpander"] summary p {
            font-size: 1.6rem !important;
            font-weight: 900 !important;
            color: #28a745 !important;
            margin: 0 !important;
        }
        div[data-testid="stExpander"] > div[role="button"] {
            font-size: 1.6rem;
            font-weight: 700;
            color: #28a745 !important;
            background: #f9fdf9 !important;
            border-radius: 14px 14px 0 0 !important;
            padding: 16px 20px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .material-symbols-outlined {
            font-size: 24px;
            color: #28a745;
            vertical-align: middle;
        }
        .streamlit-expanderContent {
            padding: 20px 25px !important;
        }
        .field-desc {
            font-size: 0.9rem;
            color: #555;
            margin-top: -8px;
            margin-bottom: 15px;
            font-style: italic;
        }
        /* Predict Button */
        div.stButton > button {
            width: 100%;
            background-color: #28a745;
            color: white;
            padding: 16px;
            font-size: 1.2rem;
            font-weight: 700;
            border-radius: 10px;
            border: none;
            box-shadow: 0px 4px 12px rgba(0,0,0,0.15);
            transition: background 0.3s, transform 0.2s;
        }
        div.stButton > button:hover {
            background-color: #218838;
            cursor: pointer;
            transform: scale(1.02);
        }
    </style>
    """, unsafe_allow_html=True)

# Function to display a custom footer
def footer():
    """Custom footer with team name and styling."""
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

# Function to display a medical disclaimer
def disclaimer():
    """Display a medical disclaimer with professional styling."""
    st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined" rel="stylesheet">
    <div style="
        background: linear-gradient(135deg, #fff3e0 0%, #ffebee 100%);
        border: 2px solid #ff9800;
        border-radius: 12px;
        padding: 25px;
        margin: 30px 0;
        box-shadow: 0 4px 12px rgba(255, 152, 0, 0.2);
        display: flex;
        flex-direction: column;
        gap: 15px;
    ">
        <div style="display: flex; align-items: center; justify-content: space-between;">
            <div style="display: flex; align-items: center;">
                <h3 style="color: #e65100; margin: 0; font-size: 1.6rem;">Medical Disclaimer</h3>
            </div>
        </div>
        <div style="color: #424242; font-size: 1.1rem; line-height: 1.6;">
            <p><strong>Important:</strong> This stroke risk prediction tool is for educational and informational purposes only. 
            The predictions are <strong>not 100% accurate</strong> and should not be used as a substitute for professional medical advice, 
            diagnosis, or treatment.</p>
            <strong>If you are worried about your stroke risk or have any health concerns, please consult with a qualified 
            healthcare professional or your doctor immediately.</strong></p>
            <p style="color: #757575; font-size: 0.95rem; font-style: italic;">
            Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)