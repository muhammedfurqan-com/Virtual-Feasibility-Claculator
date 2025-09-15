import streamlit as st
import streamlit_authenticator as stauth

# ------------------------
# Authentication Setup
# ------------------------
config = st.secrets["credentials"]

authenticator = stauth.Authenticate(
    {"usernames": {
        config["username"]: {
            "name": "Administrator",
            "password": config["password"]
        }
    }},
    "app_cookie",       # cookie name
    "abcdef",           # key for hashing cookie
    cookie_expiry_days=1
)

# Login widget
name, authentication_status, username = authenticator.login("Login", "main")

if authentication_status is False:
    st.error("Username/password is incorrect")
    st.stop()
elif authentication_status is None:
    st.warning("Please enter your username and password")
    st.stop()

# Sidebar logout
with st.sidebar:
    authenticator.logout("Logout", "sidebar")

# ------------------------
# Your main app content
# ------------------------
st.title("Virtual Feasibility Calculator")

st.write(f"Welcome **{name}** 👋, you are now logged in!")

# Example app body (replace with your calculator logic)
st.subheader("Demo Section")
st.write("This is where your feasibility calculator logic will go.")

