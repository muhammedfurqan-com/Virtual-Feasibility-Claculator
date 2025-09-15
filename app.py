# app.py
"""
Final clean version of the Nearest Location Finder app with admin authentication.
- Sidebar: App (User) / Admin
- Admin: upload backend CSV/XLSX, set feasible distance, set backend-duplicate suffix
- User: upload/paste input, parse, choose lat/lon if needed, apply filters, choose Nth,
        compute nearest matching row(s), output combined rows
"""

import streamlit as st
import pandas as pd
import numpy as np
import math
from io import StringIO
import os
import json
import io
from github import Github
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

# -------------------------
# Config helpers
# -------------------------
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

cfg = load_config()

# -------------------------
# Backend helpers
# -------------------------
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

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Nearest Site Finder", layout="wide")
st.title("Nearest Site Finder")

# -------------------------
# Authentication setup
# -------------------------
authenticator = stauth.Authenticate(
    st.secrets["credentials"],
    cookie_name=st.secrets["cookie"]["name"],
    key=st.secrets["cookie"]["key"],
    cookie_expiry_days=st.secrets["cookie"]["expiry_days"]
)

page = st.sidebar.radio("Navigation", ["App", "Admin"])

# -------------------------
# Admin page
# -------------------------
if page == "Admin":
    name, authentication_status, username = authenticator.login("Login", "main")
    if authentication_status is False:
        st.error("Username/password is incorrect")
        st.stop()
    elif authentication_status is None:
        st.warning("Please enter your username and password")
        st.stop()
    authenticator.logout("Logout", "sidebar")

    st.subheader("Upload backend master (CSV/XLSX)")
    backend_upload = st.file_uploader("Upload file (will replace current backend)", type=["csv","xlsx"])
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
# User page
# -------------------------
if page == "App":
    backend_df = load_backend_from_github()
    if backend_df is None:
        st.warning("No backend file found. Ask Admin to upload backend data.")
        st.stop()

    st.write(f"**Backend rows:** {len(backend_df)}")

    if "user_df" not in st.session_state:
        st.session_state["user_df"] = None

    st.subheader("1) Provide your input data")
    tab1, tab2 = st.tabs(["Upload file", "Paste table"])
    with tab1:
        user_file = st.file_uploader("Upload CSV/XLSX input (contains your points)", type=["csv","xlsx"], key="upload_input")
        if user_file is not None:
            try:
                udf = safe_read_table(user_file, filename=getattr(user_file, "name", None))
                udf = normalize_latlon_names(udf)
                st.session_state["user_df"] = udf
                st.success(f"Uploaded input data: {len(udf)} rows")
            except Exception as e:
                st.error(f"Failed to parse uploaded input: {e}")
    with tab2:
        pasted = st.text_area("Paste tab-separated or comma-separated data (include header)", height=200, key="paste_input")
        if st.button("Parse pasted data", key="parse_paste"):
            if not pasted.strip():
                st.warning("Please paste some data first.")
            else:
                try:
                    if "\t" in pasted:
                        udf = pd.read_csv(StringIO(pasted), sep="\t", engine="python")
                    elif "," in pasted:
                        udf = pd.read_csv(StringIO(pasted), sep=",", engine="python")
                    else:
                        udf = pd.read_csv(StringIO(pasted), delim_whitespace=True, engine="python")
                    udf = normalize_latlon_names(udf)
                    st.session_state["user_df"] = udf
                    st.success(f"Parsed pasted input: {len(udf)} rows")
                except Exception as e:
                    st.error(f"Failed to parse pasted data: {e}")

    user_df = st.session_state.get("user_df")
    if user_df is None or user_df.empty:
        st.info("Upload or paste your input to continue.")
        st.stop()

    st.subheader("Preview input (first 10 rows)")
    st.dataframe(user_df.head(10))

    # Latitude/Longitude selection
    guessed_lat, guessed_lon = detect_latlon_candidates(user_df)
    cols = list(user_df.columns)
    lat_index = cols.index(guessed_lat) if guessed_lat in cols else (0 if len(cols)>0 else None)
    lon_index = cols.index(guessed_lon) if guessed_lon in cols else (1 if len(cols)>1 else None)
    col1, col2 = st.columns(2)
    with col1:
        user_lat_col = st.selectbox("Latitude column (choose)", options=cols, index=lat_index)
    with col2:
        user_lon_col = st.selectbox("Longitude column (choose)", options=cols, index=lon_index)

# ------
# The rest of your matching logic remains exactly the same as before
# Include apply_coord_format, filters, vectorized matching, result generation, download
# ------

