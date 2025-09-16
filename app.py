import streamlit as st
import pandas as pd
from github import Github
import io

# --------------------------
# 1. GitHub persistence helpers
# --------------------------
def get_github_repo():
    g = Github(st.secrets["GITHUB_TOKEN"])
    return g.get_repo(st.secrets["GITHUB_REPO"])

def load_backend_from_github():
    repo = get_github_repo()
    try:
        contents = repo.get_contents(st.secrets["BACKEND_FILE_PATH"])

        # Decode to string explicitly
        csv_content = contents.decoded_content.decode("utf-8", errors="ignore")

        # Pass directly to pandas
        df = pd.read_csv(io.StringIO(csv_content), encoding="utf-8")

        st.success("✅ Backend data loaded successfully!")
        return df

    except Exception as e:
        st.error(f"❌ Could not load backend file: {e}")
        return pd.DataFrame()
def save_backend_to_github(df):
    repo = get_github_repo()
    csv_buf = io.StringIO()
    df.to_csv(csv_buf, index=False)
    csv_content = csv_buf.getvalue()

    try:
        contents = repo.get_contents(st.secrets["BACKEND_FILE_PATH"])
        repo.update_file(
            path=contents.path,
            message="Update backend file",
            content=csv_content,
            sha=contents.sha,
        )
    except Exception:
        repo.create_file(
            path=st.secrets["BACKEND_FILE_PATH"],
            message="Create backend file",
            content=csv_content,
        )

# --------------------------
# 2. App Layout
# --------------------------
st.set_page_config(page_title="Nearest Site Finder", layout="wide")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["User Page", "Admin Page"])

# --------------------------
# 3. User Page
# --------------------------
if page == "User Page":
    st.header("📍 Nearest Site Finder")

    df = load_backend_from_github()
    if df.empty:
        st.info("No site data available. Please ask admin to upload.")
    else:
        st.write("Available sites:", df.head())
        st.success("✅ Site finder logic goes here!")

# --------------------------
# 4. Admin Page
# --------------------------
elif page == "Admin Page":
    st.header("🔑 Admin Panel")

    st.subheader("Upload Backend File")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            save_backend_to_github(df)
            st.success("✅ Backend file saved to GitHub!")
        except Exception as e:
            st.error(f"Error uploading file: {e}")
