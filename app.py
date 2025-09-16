# app.py
import streamlit as st
import pandas as pd
from github import Github
import io
import os
import tempfile

# --------------------------
# GitHub helpers
# --------------------------
def get_github_repo():
    try:
        g = Github(st.secrets["GITHUB_TOKEN"])
        return g.get_repo(st.secrets["GITHUB_REPO"])
    except Exception as e:
        st.error(f"❌ GitHub access error (check GITHUB_TOKEN / GITHUB_REPO in secrets): {e}")
        return None

def save_backend_to_github(df):
    """Save dataframe to GitHub as UTF-8 CSV (overwrites or creates file)."""
    repo = get_github_repo()
    if repo is None:
        st.error("No repo available to save file.")
        return False

    csv_buf = io.StringIO()
    # Force UTF-8 encoding when writing string content
    df.to_csv(csv_buf, index=False)
    csv_content = csv_buf.getvalue()  # string

    try:
        contents = repo.get_contents(st.secrets["BACKEND_FILE_PATH"])
        repo.update_file(path=contents.path, message="Update backend file", content=csv_content, sha=contents.sha)
        st.success("✅ Backend updated in GitHub (update_file).")
        return True
    except Exception as e_update:
        # Try create_file if update failed (file doesn't exist)
        try:
            repo.create_file(path=st.secrets["BACKEND_FILE_PATH"], message="Create backend file", content=csv_content)
            st.success("✅ Backend created in GitHub (create_file).")
            return True
        except Exception as e_create:
            st.error(f"❌ Failed to save backend to GitHub. update error: {e_update} ; create error: {e_create}")
            return False

def load_backend_from_github():
    """Robust loader: try multiple decodings and read methods, with debug output."""
    repo = get_github_repo()
    if repo is None:
        return pd.DataFrame()

    path = st.secrets["BACKEND_FILE_PATH"]
    try:
        contents = repo.get_contents(path)
    except Exception as e:
        st.error(f"❌ Could not fetch file at '{path}' from GitHub: {e}")
        return pd.DataFrame()

    raw = contents.decoded_content  # often bytes, sometimes other

    # Debug: type + length
    try:
        st.write(f"DEBUG: contents type={type(raw)}, length={len(raw)}")
    except Exception:
        st.write(f"DEBUG: contents type={type(raw)}")

    # If it is already a str, attempt to read directly
    if isinstance(raw, str):
        try:
            df = pd.read_csv(io.StringIO(raw), engine="python")
            st.success("✅ Loaded backend (raw str -> pandas).")
            return df
        except Exception as e:
            st.write(f"DEBUG: failed to read from str: {e}")

    # If bytes - try multiple decodings
    if isinstance(raw, (bytes, bytearray)):
        encodings_to_try = ["utf-8", "utf-8-sig", "latin1", "cp1252", "utf-16"]
        for enc in encodings_to_try:
            try:
                text = raw.decode(enc)
                st.write(f"DEBUG: decoded using {enc} (len={len(text)})")
                try:
                    df = pd.read_csv(io.StringIO(text), engine="python")
                    st.success(f"✅ Loaded backend using encoding '{enc}'.")
                    return df
                except Exception as e_read:
                    st.write(f"DEBUG: pd.read_csv failed for encoding {enc}: {e_read}")
            except Exception as e_dec:
                st.write(f"DEBUG: decode with {enc} failed: {e_dec}")

        # Next try: read bytes directly (pandas can sometimes handle from BytesIO)
        try:
            st.write("DEBUG: Trying pd.read_csv from BytesIO with engine='python' (no explicit encoding)...")
            df = pd.read_csv(io.BytesIO(raw), engine="python")
            st.success("✅ Loaded backend from BytesIO.")
            return df
        except Exception as e_bytes:
            st.write(f"DEBUG: pd.read_csv(BytesIO) failed: {e_bytes}")

    # Fallback: write raw to a temporary file and let pandas try various encodings
    try:
        tmpdir = tempfile.mkdtemp()
        tmp_path = os.path.join(tmpdir, "backend_tmp.csv")
        st.write(f"DEBUG: writing raw to temp file {tmp_path} for additional attempts.")
        if isinstance(raw, (bytes, bytearray)):
            with open(tmp_path, "wb") as f:
                f.write(raw)
        else:
            with open(tmp_path, "w", encoding="utf-8", errors="ignore") as f:
                f.write(str(raw))

        # Try reading with a set of encodings
        for enc in ["utf-8", "utf-8-sig", "latin1", "cp1252", "utf-16"]:
            try:
                df = pd.read_csv(tmp_path, encoding=enc, engine="python")
                st.success(f"✅ Loaded backend from temp file using encoding '{enc}'.")
                # cleanup not strictly necessary; OS will reclaim
                return df
            except Exception as e_tmp:
                st.write(f"DEBUG: temp-file read failed with encoding {enc}: {e_tmp}")

    except Exception as e_tmpall:
        st.write(f"DEBUG: writing/reading temp file failed: {e_tmpall}")

    # If we reach here, show some raw bytes preview to help debugging
    try:
        if isinstance(raw, (bytes, bytearray)):
            sample = bytes(raw[:200])
            st.text("DEBUG: first 200 bytes (repr):")
            st.text(repr(sample))
        else:
            st.text("DEBUG: raw content (first 200 chars):")
            st.text(str(raw)[:200])
    except Exception:
        pass

    st.error("❌ All attempts to load backend_data.csv failed. Please inspect file in GitHub (raw view) and ensure it's a valid CSV text file (UTF-8 or latin1).")
    return pd.DataFrame()

# --------------------------
# App Layout
# --------------------------
st.set_page_config(page_title="Nearest Site Finder", layout="wide")
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["User Page", "Admin Page"])

# --------------------------
# User Page
# --------------------------
if page == "User Page":
    st.header("📍 Nearest Site Finder")

    df = load_backend_from_github()
    if df.empty:
        st.info("No site data available. Please ask admin to upload.")
    else:
        st.write("Available sites (preview):")
        st.dataframe(df.head(10))
        st.success("✅ Backend loaded — you can now run the site-finder logic here.")

# --------------------------
# Admin Page
# --------------------------
elif page == "Admin Page":
    st.header("🔑 Admin Panel")
    st.subheader("Upload Backend File (CSV)")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        try:
            # Read uploaded file into a DataFrame (try utf-8 then latin1)
            try:
                df = pd.read_csv(uploaded_file, encoding="utf-8")
            except Exception:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding="latin1")

            ok = save_backend_to_github(df)
            if ok:
                st.success("✅ Backend file saved to GitHub!")
                st.write("Preview of uploaded file:")
                st.dataframe(df.head(10))
            else:
                st.error("❌ Failed to save file to GitHub.")
        except Exception as e:
            st.error(f"Error parsing/uploading file: {e}")
