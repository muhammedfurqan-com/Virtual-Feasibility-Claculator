# app.py
"""
Nearest Location Finder
- Sidebar: App (User) / Admin
- Admin: upload backend CSV/XLSX, set feasible distance, set backend conflict suffix
- User: upload/paste input, choose lat/lon if needed, apply filters, compute nearest row(s)
"""

import streamlit as st
import pandas as pd
import numpy as np
import math
from io import StringIO
import os
import json
from github import Github
import io
import streamlit_authenticator as stauth

# -------------------------
# GitHub setup
# -------------------------
token = st.secrets["GITHUB_TOKEN"]
repo_name = st.secrets["GITHUB_REPO"]
file_path = st.secrets["BACKEND_FILE_PATH"]

g = Github(token)
repo = g.get_repo(repo_name)

# -------------------------
# Config / filenames
# -------------------------
CONFIG_FILE = "app_config.json"
DEFAULT_CONFIG = {
    "feasible_km": 20.0,
    "backend_conflict_suffix": "_matched"
}

# Load / save config
def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return DEFAULT_CONFIG.copy()
    return DEFAULT_CONFIG.copy()

def save_config(cfg):
    with open(CONFIG_FILE, "w") as f:
        json.dump(cfg, f)

# Load backend from GitHub
def load_backend_from_github():
    try:
        contents = repo.get_contents(file_path)
        data = contents.decoded_content.decode()
        df = pd.read_csv(io.StringIO(data))
        df = normalize_latlon_names(df)
        return df
    except Exception as e:
        st.warning(f"Could not load backend file: {e}")
        return None

def save_backend_to_github(df):
    csv_bytes = df.to_csv(index=False).encode()
    try:
        contents = repo.get_contents(file_path)
        repo.update_file(contents.path, "Update backend data", csv_bytes, contents.sha)
        st.success("✅ Backend updated in GitHub!")
    except Exception:
        repo.create_file(file_path, "Create backend data", csv_bytes)
        st.success("✅ Backend created in GitHub!")

cfg = load_config()

# -------------------------
# Helpers
# -------------------------
def safe_read_table(obj, filename=None):
    name = (getattr(obj, "name", None) or filename or "").lower()
    try:
        if name.endswith((".xlsx", ".xls")):
            return pd.read_excel(obj, engine="openpyxl")
    except Exception as e:
        st.error(f"Excel read failed: {e}")
    try:
        return pd.read_csv(obj, sep=None, engine="python", skipinitialspace=True)
    except Exception as e:
        st.error(f"CSV read failed: {e}")
        raise

def normalize_latlon_names(df):
    col_map = {}
    for col in df.columns:
        low = col.strip().lower()
        if ("latitute" in low) or ("latitude" in low) or low == "lat" or (low.startswith("lat") and not low.startswith("platform")):
            col_map[col] = "Latitude"
            continue
        if ("longitude" in low) or low in ("lon", "lng", "long") or low.startswith("lon") or low.startswith("long"):
            col_map[col] = "Longitude"
            continue
    if col_map:
        df = df.rename(columns=col_map)
    return df

def detect_latlon_candidates(df):
    lat, lon = None, None
    for c in df.columns:
        k = "".join(ch.lower() for ch in str(c).strip() if ch.isalnum())
        if lat is None and k in ("lat", "latitude", "latitute", "latdeg", "latit"):
            lat = c
        if lon is None and k in ("lon", "longitude", "lng", "long", "longit"):
            lon = c
    if lat is None:
        for c in df.columns:
            if "lat" in c.lower() and "latitude" not in c.lower():
                lat = c
                break
    if lon is None:
        for c in df.columns:
            if "lon" in c.lower() or "lng" in c.lower():
                lon = c
                break
    return lat, lon

def vectorized_haversine(lat1, lon1, backend_lat_rad, backend_lon_rad):
    R = 6371.0
    lat1r = math.radians(lat1)
    lon1r = math.radians(lon1)
    dlat = backend_lat_rad - lat1r
    dlon = backend_lon_rad - lon1r
    a = np.sin(dlat/2.0)**2 + np.cos(lat1r)*np.cos(backend_lat_rad)*np.sin(dlon/2.0)**2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return R * c

def find_nth_index(distances, n):
    if len(distances) == 0:
        return None
    order = np.argsort(distances)
    if n <= 0:
        n = 1
    if n > len(order):
        return None
    return int(order[n-1])

def backend_column_merge_dict(backend_row, input_cols, suffix):
    mapping = {}
    for c in backend_row.index:
        if c in input_cols:
            mapping[c] = f"{c}{suffix}"
        else:
            mapping[c] = c
    return mapping

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Nearest Site Finder", layout="wide")
st.title("Nearest Site Finder")

# -------------------------
# Authentication setup (admin only)
# -------------------------
credentials = {
    "usernames": {
        "admin": {
            "name": st.secrets["credentials"]["usernames"]["admin"]["name"],
            "password": st.secrets["credentials"]["usernames"]["admin"]["password"]
        }
    }
}

authenticator = stauth.Authenticate(
    credentials=credentials,
    cookie_name=st.secrets["cookie"]["name"],
    key=st.secrets["cookie"]["key"],
    cookie_expiry_days=int(st.secrets["cookie"]["expiry_days"])
)

# Admin login widget
name, authentication_status, username = authenticator.login(
    label="Admin Login",
    location="sidebar"
)

# Handle authentication
if authentication_status is False:
    st.error("Username/password is incorrect")
    st.stop()
elif authentication_status is None:
    st.warning("Please enter your username and password")
    st.stop()

# Logout button in sidebar
with st.sidebar:
    authenticator.logout("Logout", "sidebar")

# Sidebar navigation
page = st.sidebar.radio("Navigation", ["App", "Admin"])

# -------------------------
# Admin page
# -------------------------
if page == "Admin":
    st.subheader("Admin Panel")
    
    backend_upload = st.file_uploader("Upload backend master (CSV/XLSX)", type=["csv","xlsx"])
    if backend_upload is not None:
        try:
            df = safe_read_table(backend_upload, filename=getattr(backend_upload, "name", None))
            df = normalize_latlon_names(df)
            save_backend_to_github(df)
        except Exception as e:
            st.error(f"Failed to read and save backend: {e}")

    bdf = load_backend_from_github()
    if bdf is not None:
        st.subheader("Current backend preview (first 10 rows)")
        st.dataframe(bdf.head(10))
    else:
        st.info("No backend file found. Upload above.")

    st.markdown("---")
    st.subheader("Settings")
    new_feasible = st.number_input("Feasible distance (km)", min_value=0.1, value=float(cfg.get("feasible_km",20.0)))
    new_suffix = st.text_input("Suffix to add to backend columns that conflict with input names", value=cfg.get("backend_conflict_suffix","_matched"))
    if st.button("Save settings"):
        cfg["feasible_km"] = float(new_feasible)
        cfg["backend_conflict_suffix"] = new_suffix.strip() or "_matched"
        save_config(cfg)
        st.success("Settings saved to app_config.json")

# -------------------------
# App (User) page
# -------------------------
if page == "App":
    backend_df = load_backend_from_github()
    if backend_df is None:
        st.warning("No backend file found in GitHub. Ask Admin to upload backend data.")
        st.stop()

    st.write(f"**Backend rows:** {len(backend_df)}")
    # ... rest of your user page logic here ...
