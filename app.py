import streamlit as st
import pandas as pd

# --------------------------
# 1. App Layout
# --------------------------
st.set_page_config(page_title="Nearest Site Finder", layout="wide")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["User Page", "Admin Page"])

# --------------------------
# 2. Global In-Memory Storage
# --------------------------
# This will reset every time the app reloads
if "backend_df" not in st.session_state:
    st.session_state["backend_df"] = pd.DataFrame()

# --------------------------
# 3. User Page
# --------------------------
if page == "User Page":
    st.header("📍 Nearest Site Finder")

    df = st.session_state["backend_df"]
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
            st.session_state["backend_df"] = df
            st.success("✅ Backend file uploaded and stored in memory!")
        except Exception as e:
            st.error(f"Error uploading file: {e}")
