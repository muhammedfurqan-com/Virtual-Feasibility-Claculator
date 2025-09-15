import streamlit as st
import pandas as pd
import numpy as np
import math
from io import BytesIO
from github import Github
import streamlit_authenticator as stauth

import streamlit as st
import streamlit_authenticator as stauth

# Load configuration from secrets.toml
config = {
    "credentials": {
        "usernames": {
            "admin": {
                "name": st.secrets["credentials"]["usernames"]["admin"]["name"],
                "password": st.secrets["credentials"]["usernames"]["admin"]["password"],
            }
        }
    },
    "cookie": {
        "expiry_days": st.secrets["cookie"]["expiry_days"],
        "key": st.secrets["cookie"]["key"],
        "name": st.secrets["cookie"]["name"],
    },
    "preauthorized": {
        "emails": st.secrets["preauthorized"]["emails"],
    },
}

authenticator = stauth.Authenticate(
    config["credentials"],
    config["cookie"]["name"],
    config["cookie"]["key"],
    config["cookie"]["expiry_days"],
    config["preauthorized"],
)

# ================================================================
# GitHub Setup
# ================================================================
GITHUB_TOKEN = st.secrets["GITHUB_TOKEN"]
GITHUB_REPO = st.secrets["GITHUB_REPO"]
BACKEND_FILE_PATH = st.secrets["BACKEND_FILE_PATH"]

g = Github(GITHUB_TOKEN)
repo = g.get_repo(GITHUB_REPO)


# ================================================================
# Utility Functions
# ================================================================
def load_backend_from_github():
    try:
        file_content = repo.get_contents(BACKEND_FILE_PATH)
        df = pd.read_csv(BytesIO(file_content.decoded_content))
        return df
    except Exception as e:
        st.error(f"Error loading backend file: {e}")
        return pd.DataFrame()


def save_backend_to_github(df):
    try:
        content = df.to_csv(index=False).encode()
        try:
            file_content = repo.get_contents(BACKEND_FILE_PATH)
            repo.update_file(
                BACKEND_FILE_PATH,
                "Update backend data",
                content,
                file_content.sha,
            )
        except Exception:
            repo.create_file(
                BACKEND_FILE_PATH,
                "Create backend data",
                content,
            )
    except Exception as e:
        st.error(f"Error saving backend file: {e}")


# ================================================================
# Distance Calculation
# ================================================================
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def find_nearest_sites(df, lat, lon, n=5):
    df["Distance_km"] = df.apply(
        lambda row: haversine(lat, lon, row["Latitude"], row["Longitude"]),
        axis=1,
    )
    return df.nsmallest(n, "Distance_km")


# ================================================================
# Authentication Setup (Admin Only)
# ================================================================
#usernames = st.secrets["credentials"]["usernames"]
#passwords = st.secrets["credentials"]["passwords"]

credentials = {
    "usernames": {
        usernames[0]: {
            "name": "Admin",
            "password": passwords[0],
        }
    }
}

authenticator = stauth.Authenticate(
    credentials,
    "nearest_site_app",
    "auth",
    cookie_expiry_days=1,
)


# ================================================================
# Streamlit UI
# ================================================================
st.set_page_config(page_title="Nearest Site Finder", layout="wide")

st.title("📡 Nearest Site Finder App")

menu = ["App", "Admin"]
choice = st.sidebar.radio("Navigation", menu)

# ------------------------------------------------
# App Page (Public)
# ------------------------------------------------
if choice == "App":
    st.header("Find Nearest Sites")

    df_backend = load_backend_from_github()

    if df_backend.empty:
        st.warning("No backend data available. Please ask admin to upload it.")
    else:
        lat = st.number_input("Enter Latitude", format="%.6f")
        lon = st.number_input("Enter Longitude", format="%.6f")
        n = st.slider("Number of nearest sites", 1, 20, 5)

        if st.button("Find Nearest"):
            result = find_nearest_sites(df_backend.copy(), lat, lon, n)
            st.dataframe(result)


# ------------------------------------------------
# Admin Page (Protected)
# ------------------------------------------------
elif choice == "Admin":
    name, authentication_status, username = authenticator.login()
   # name, authentication_status, username = authenticator.login("Login", "main")

    if authentication_status:
        st.success(f"Welcome {name}! You are logged in as admin.")
        st.header("Admin Dashboard")

        df_backend = load_backend_from_github()
        st.subheader("Current Backend Data")
        st.dataframe(df_backend)

        uploaded_file = st.file_uploader("Upload new backend CSV", type=["csv"])
        if uploaded_file is not None:
            df_new = pd.read_csv(uploaded_file)
            save_backend_to_github(df_new)
            st.success("Backend file updated successfully!")

        if st.button("Logout"):
            authenticator.logout("Logout", "main")

    elif authentication_status is False:
        st.error("Invalid username or password")
    else:
        st.warning("Please log in to access this page.")
