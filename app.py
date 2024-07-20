import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import brain_tumor
import pneumonia
from util import set_background, set_sidebar_background

def login():
    placeholder = st.empty()

    actual_user_name = "user"
    actual_password = "12345"

    # Insert a form in the container
    with placeholder.form("login"):
        st.markdown("<h3 style='color: black;'>Enter your credentials</h3>", unsafe_allow_html=True)
        user_name = st.text_input("User Name")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")

    if submit and user_name == actual_user_name and password == actual_password:
        st.session_state["logged_in"] = True
        placeholder.empty()
    elif submit:
        st.error("Login failed")

# Initial page setup
st.set_page_config(
    page_title="Medical Image Analysis",
    page_icon="👨‍⚕️",
)

# Set background images
set_background('C:\\Users\\Starlord\\Downloads\\gf.jpeg')
set_sidebar_background('C:\\Users\\Starlord\\Downloads\\gf.jpeg')

# Website header
st.markdown("<h1 style='color: black;'>🩺 Welcome to our Medical Image Analysis website</h1>", unsafe_allow_html=True)

# Login Page
if "logged_in" not in st.session_state:
    login()

# Choose prediction type
if "logged_in" in st.session_state:
    st.sidebar.markdown("<h2 style='color: red;'>🧬 Select Disease Detection Option</h2>", unsafe_allow_html=True)

    st.markdown("""
    <style>
    .stSelectbox {
        background-color: black;
        color: black;
        border: 2px solid black;
    }
    .stSelectbox div[data-baseweb="select"] span {
        color: black; /* Default option color */
    }
    </style>
    """, unsafe_allow_html=True)

    option = st.sidebar.selectbox("Disease Name", ["Brain Tumor Detection", "Pneumonia Detection"], key="option1")

    if option == "Brain Tumor Detection":
        brain_tumor.prediction()
    elif option == "Pneumonia Detection":
        pneumonia.prediction()
