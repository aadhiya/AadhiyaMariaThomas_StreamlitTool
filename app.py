# app.py
import streamlit as st
import polars as pl
import pandas as pd
import io

# PAGE CONFIGURATION
st.set_page_config(
    page_title="Streamlit based Data Cleansing, Profiling & ML Tool by Aadhiya Maria Thomas",
    layout="wide"
)

# APP TITLE 
st.title("Streamlit Data Tool: Cleansing, Profiling & Prdictive ML Models by Aadhiya Maria Thomas")
st.markdown("""
Upload a CSV or Excel file to begin.  
This first step loads your dataset and shows a quick preview.
""")

# FILE UPLOADER 
uploaded_file = st.file_uploader(
    "Upload your dataset (CSV or Excel)",
    type=["csv", "xlsx"],
    help="Select a .csv or .xlsx file from your computer"
)

# PREVIEW 
if uploaded_file is not None:
    try:
        # Check if it's CSV or Excel
        if uploaded_file.name.endswith(".csv"):
            # Read with Polars for speed
            df = pl.read_csv(uploaded_file)
        else:  # Excel
            df = pl.from_pandas(pd.read_excel(uploaded_file))

        # Show basic info
        st.subheader("Dataset Overview")
        st.write(f"**Shape:** {df.height} rows × {df.width} columns")
        st.write(f"**File name:** {uploaded_file.name}")

        # Show first 10 rows
        st.subheader("Data Preview")
        st.dataframe(df.head(10))

        # Show column types
        st.subheader("Column Types")
        st.write(df.dtypes)

    except Exception as e:
        st.error(f"Error loading file: {e}")
else:
    st.info("👆 Please upload a CSV or Excel file to see preview.")
