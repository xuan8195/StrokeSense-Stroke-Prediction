import streamlit as st

# Function to define the theme and sidebar layout
def theme():
    """
    Sets up the sidebar layout and styling for the application.
    Includes useful links and resources for stroke awareness and prevention.
    """
    
    # Sidebar header and description
    st.sidebar.markdown("""
        <style>
            /* Styling for the sidebar title */
            .sidebar-title {
                font-size: 1.3rem;
                font-weight: 700;
                color: #1a3c40;
                margin-bottom: 5px;
            }
            /* Styling for the sidebar subtext */
            .sidebar-subtext {
                font-size: 0.95rem;
                color: #555;
                margin-bottom: 15px;
            }
            /* Hover effect for resource cards */
            .link-card:hover {
                background: #e9f2f2;
                box-shadow: 0 2px 6px rgba(0,0,0,0.08);
            }
            /* Styling for resource cards */
            .link-card {
                background: #ffffff;
                padding: 10px 15px;
                border-radius: 8px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.08);
                margin-bottom: 10px;
            }
            /* Styling for links inside resource cards */
            .link-card a {
                font-weight: 600;
                text-decoration: none;
                color: #1a3c40;
            }
            .link-card a:hover {
                color: #0f5c5c;
            }
            /* Styling for descriptions inside resource cards */
            .link-card p {
                margin: 5px 0 0;
                font-size: 0.85rem;
                color: #555;
            }
        </style>
        <div class="sidebar-title">Useful Links & Resources</div>
        <div class="sidebar-subtext">Resources to help you understand stroke risk and prevention in Singapore:</div>
    """, unsafe_allow_html=True)

    # List of useful resources displayed as cards in the sidebar
    resources = [
        ("HealthHub – Stroke Hub", "https://www.healthhub.sg/programmes/strokehub", "Comprehensive guide on stroke awareness & prevention."),
        ("Singapore Heart Foundation", "https://www.myheart.org.sg/health/heart-conditions/stroke/", "Information on stroke causes, prevention, and wellness programs."),
        ("Singapore National Stroke Association (SNSA)", "https://www.snsasg.org/", "Support for stroke survivors & caregivers."),
        ("Health Promotion Board (HPB)", "https://www.hpb.gov.sg", "Health campaigns & screening programs under Healthier SG."),
        ("St. Luke’s Hospital", "https://www.slh.org.sg/how-to-reduce-my-risk-of-getting-stroke/", "Rehabilitation support and stroke prevention programs.")
    ]

    # Loop through resources and display them as cards
    for name, link, desc in resources:
        st.sidebar.markdown(f"""
            <div class="link-card">
                <a href="{link}" target="_blank">{name}</a>
                <p>{desc}</p>
            </div>
        """, unsafe_allow_html=True)

    # Inject CSS for sidebar styling
    st.markdown("""
        <style>
        /* Styling for sidebar text and header */
        [data-testid="stSidebar"] ul, 
        [data-testid="stSidebarNavItems"], 
        [data-testid="stSidebarNavLink"] {
            font-size: 2rem !important;
            font-weight: 600 !important;
        }
        [data-testid="stSidebarHeader"] {
            font-size: 2rem !important;
            font-weight: 700 !important;
        }
        </style>
    """, unsafe_allow_html=True)

    # Styling for buttons in the application
    st.markdown("""
    <link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
     <style>
        div.stButton > button {
            width: 100%;
            background-color:#5298D2;
            color: white;
            padding: 14px 20px;
            font-size: 1.2rem;
            font-weight: 900;
            border-radius: 10px;
            border: none;
            margin-bottom: 8px;
            box-shadow: 0px 4px 8px rgba(0,0,0,0.1);
            cursor: pointer;
            transition: all 0.3s ease;
        }
        div.stButton > button:hover {
            background-color: #4682B4 ;
            transform: translateY(-2px);
            box-shadow: 0px 6px 12px rgba(0,0,0,0.2);
            cursor: pointer;
            color: white;
        }
    """, unsafe_allow_html=True)

# Function to set the background gradient animation for the app
def app_background():
    """
    Adds a gradient background animation to the application.
    Provides a visually appealing and dynamic background effect.
    """
    st.markdown("""
        <style>
            .stApp {
                background: linear-gradient(-45deg, #E8F8FA, #ffffff, #F5FCFD, #ffffff);
                background-size: 400% 400%;
                animation: gradientBG 15s ease infinite;
            }
            @keyframes gradientBG {
                0% {background-position: 0% 50%;}
                50% {background-position: 100% 50%;}
                100% {background-position: 0% 50%;}
            }
        </style>
    """, unsafe_allow_html=True)


