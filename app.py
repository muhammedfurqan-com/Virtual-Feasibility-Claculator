# app.py
"""
Nearest Site Finder App
- Admin login enabled (admin/admin)
- Users can access app without login
- GitHub backend load/save
"""

import streamlit as st
import pandas as pd
import numpy as np
import math
from io import StringIO
import os
import json
import streamlit_authenticator as stauth
from github import Github
import io

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

def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                return json.load(f)
        except:
            return DEFAULT_CONFIG.copy()
    return DEFAULT_CONFIG.copy()

def save_config(cfg):
    with open(CONFIG_FILE, "w") as f:
        json.dump(cfg, f)

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
    except:
        pass
    return pd.read_csv(obj, sep=None, engine="python", skipinitialspace=True)

def normalize_latlon_names(df):
    col_map = {}
    for col in df.columns:
        low = col.strip().lower()
        if "lat" in low:
            col_map[col] = "Latitude"
        if "lon" in low:
            col_map[col] = "Longitude"
    return df.rename(columns=col_map)

def detect_latlon_candidates(df):
    lat, lon = None, None
    for c in df.columns:
        k = "".join(ch.lower() for ch in str(c).strip() if ch.isalnum())
        if lat is None and k in ("lat", "latitude", "latitute"):
            lat = c
        if lon is None and k in ("lon", "longitude", "lng", "long"):
            lon = c
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
# Authentication (Admin only)
# -------------------------
authenticator = stauth.Authenticate(
    st.secrets["credentials"],
    cookie_name=st.secrets["cookie"]["name"],
    key=st.secrets["cookie"]["key"],
    cookie_expiry_days=st.secrets["cookie"]["expiry_days"]
)

# Admin login status
name, authentication_status, username = authenticator.login("Login", "main")

# Sidebar navigation
page = st.sidebar.radio("Navigation", ["App", "Admin"])

# -------------------------
# Admin page
# -------------------------
if page == "Admin":
    if authentication_status != True:
        st.error("Admin login required")
        st.stop()

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
    if "user_df" not in st.session_state:
        st.session_state["user_df"] = None

    # Input
    st.subheader("Upload input data")
    user_file = st.file_uploader("Upload CSV/XLSX input", type=["csv","xlsx"], key="upload_input")
    if user_file is not None:
        try:
            udf = safe_read_table(user_file, filename=getattr(user_file, "name", None))
            udf = normalize_latlon_names(udf)
            st.session_state["user_df"] = udf
            st.success(f"Uploaded input data: {len(udf)} rows")
        except Exception as e:
            st.error(f"Failed to parse uploaded input: {e}")

    user_df = st.session_state.get("user_df")
    if user_df is None or user_df.empty:
        st.info("Upload input data to continue.")
        st.stop()

    st.subheader("Preview input (first 10 rows)")
    st.dataframe(user_df.head(10))

    # Lat/Lon detection
    guessed_lat, guessed_lon = detect_latlon_candidates(user_df)
    cols = list(user_df.columns)
    lat_index = cols.index(guessed_lat) if guessed_lat in cols else 0
    lon_index = cols.index(guessed_lon) if guessed_lon in cols else 1
    col1, col2 = st.columns(2)
    with col1:
        user_lat_col = st.selectbox("Latitude column", options=cols, index=lat_index)
    with col2:
        user_lon_col = st.selectbox("Longitude column", options=cols, index=lon_index)

    st.subheader("Matching options")
    global_nth = st.number_input("Global Nth nearest", min_value=1, value=1)
    distance_unit = st.radio("Distance unit", ("Kilometers","Miles"), horizontal=True)
    conflict_suffix = cfg.get("backend_conflict_suffix", "_matched")

    # Run matching
    if st.button("Run matching"):
        df_user = user_df.copy()
        df_user[user_lat_col] = pd.to_numeric(df_user[user_lat_col], errors="coerce")
        df_user[user_lon_col] = pd.to_numeric(df_user[user_lon_col], errors="coerce")
        df_user = df_user.dropna(subset=[user_lat_col,user_lon_col]).reset_index(drop=True)
        if df_user.empty:
            st.error("No valid input rows to process.")
            st.stop()

        backend_filtered = backend_df.copy()
        backend_lat_rad = np.radians(backend_filtered["Latitude"].to_numpy(dtype=float))
        backend_lon_rad = np.radians(backend_filtered["Longitude"].to_numpy(dtype=float))

        results_frames = []
        input_columns = list(df_user.columns)
        for idx, row in df_user.iterrows():
            lat_val = float(row[user_lat_col])
            lon_val = float(row[user_lon_col])
            dists = vectorized_haversine(lat_val, lon_val, backend_lat_rad, backend_lon_rad)
            j = find_nth_index(dists, global_nth)
            if j is None:
                empty_backend = {c: None for c in backend_filtered.columns}
                empty_backend.update({"Distance_km": None, "Distance_miles": None, "Feasible": None})
                backend_row = pd.Series(empty_backend)
            else:
                backend_row = backend_filtered.iloc[int(j)].copy()
                dist_km = float(dists[j])
                backend_row["Distance_km"] = round(dist_km, 6)
                backend_row["Distance_miles"] = round(dist_km * 0.621371, 6)
                backend_row["Feasible"] = "Feasible" if dist_km <= float(cfg.get("feasible_km",20)) else "Not Feasible"

            merge_map = backend_column_merge_dict(backend_row, input_columns, conflict_suffix)
            backend_row_renamed = backend_row.rename(index=merge_map).to_frame().T.reset_index(drop=True)
            input_row_df = row.to_frame().T.reset_index(drop=True)

            combined = pd.concat([input_row_df, backend_row_renamed], axis=1)
            combined["Nth_used"] = global_nth
            results_frames.append(combined)

        final = pd.concat(results_frames, ignore_index=True)
        st.success("Matching completed.")
        st.subheader("Results (first 200 rows)")
        st.dataframe(final.head(200), use_container_width=True)
        csv_bytes = final.to_csv(index=False).encode("utf-8")
        st.download_button("Download results CSV", data=csv_bytes, file_name="nearest_results.csv", mime="text/csv")
