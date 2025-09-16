import streamlit as st
import pandas as pd
import streamlit_authenticator as stauth
from github import Github
import io

# --------------------------
# 1. Authentication config
# --------------------------
credentials = {
    "usernames": {
        "admin": {
            "name": st.secrets["credentials"]["usernames"]["admin"]["name"],
            "password": st.secrets["credentials"]["usernames"]["admin"]["password"],
        }
    }
}

authenticator = stauth.Authenticate(
    credentials,
    st.secrets["cookie"]["name"],      # cookie name
    st.secrets["cookie"]["key"],       # signature key
    cookie_expiry_days=st.secrets["cookie"]["expiry_days"]
)

# --------------------------
# 2. GitHub persistence helpers
# --------------------------
def get_github_repo():
    g = Github(st.secrets["GITHUB_TOKEN"])
    return g.get_repo(st.secrets["GITHUB_REPO"])

def load_backend_from_github():
    repo = get_github_repo()
    try:
        contents = repo.get_contents(st.secrets["BACKEND_FILE_PATH"])
        return pd.read_csv(io.BytesIO(contents.decoded_content))
    except Exception:
        st.warning("⚠️ No backend file found yet. Please upload from Admin page.")
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
# 3. App Layout
# --------------------------
st.set_page_config(page_title="Nearest Site Finder", layout="wide")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["User Page", "Admin Page"])

# --------------------------
# 4. User Page (no login required)
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
# 5. Admin Page (login required)
# --------------------------
elif page == "Admin Page":
    st.header("🔑 Admin Panel")

    name, authentication_status, username = authenticator.login("Admin Login", "main")

    if authentication_status:
        st.success(f"Welcome, {name}!")

        st.subheader("Upload Backend File")
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                save_backend_to_github(df)
                st.success("✅ Backend file saved to GitHub!")
            except Exception as e:
                st.error(f"Error uploading file: {e}")

        authenticator.logout("Logout", "sidebar")

    elif authentication_status is False:
        st.error("❌ Invalid username or password")

    elif authentication_status is None:
        st.warning("Please enter your username and password")
