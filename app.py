import streamlit as st
import polars as pl
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import io
import locale
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_squared_error, r2_score
import os
import boto3
#port = int(os.environ.get("PORT", 8501))
# ============================= PAGE CONFIG =============================
st.set_page_config(
    page_title="Aadhiya Maria Thomas - Streamlit Data Tool",
    layout="wide"
)
st.title("📊 Streamlit Data Tool: Cleansing, Profiling & ML by Aadhiya Thomas")

numeric_polars_types = [
    pl.Int8, pl.Int16, pl.Int32, pl.Int64,
    pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
    pl.Float32, pl.Float64
]

# ============================= SIDEBAR =============================
bucket_name = "aadhiya-streamlit-data"  # Your S3 bucket name

st.sidebar.header("📂 S3 Data")

uploaded_file = st.sidebar.file_uploader("Upload CSV/Excel to S3", type=["csv", "xlsx"])
if uploaded_file is not None:
    st.info(f"Received uploaded file: name={uploaded_file.name}, size={uploaded_file.size} bytes")
    try:
        st.write("Initializing boto3 client...")
        s3 = boto3.client('s3')
        st.write(f"Attempting upload to bucket: {bucket_name} as key: {uploaded_file.name}")
        response = s3.upload_fileobj(uploaded_file, bucket_name, uploaded_file.name)
        st.success(f"File '{uploaded_file.name}' uploaded to S3 bucket '{bucket_name}' successfully!")
    except Exception as e:
        st.error(f"Upload failed: {e}")
        st.write("Exception details:", str(e))

# Optional: List files in S3 for selection (add debug here as well)
def list_s3_files(bucket):
    try:
        # Remove these debug prints:
        # st.write("Listing files in S3 bucket:", bucket)
        s3 = boto3.client('s3')
        response = s3.list_objects_v2(Bucket=bucket)
        # Remove: st.write("S3 list_objects_v2 response:", response)
        if 'Contents' in response:
            return [obj['Key'] for obj in response['Contents']]
        return []
    except Exception as e:
        st.error(f"Failed to list S3 files: {e}")
        # Optionally keep this for troubleshooting, but remove in production:
        # st.write("Exception details:", str(e))
        return []

s3_files = list_s3_files(bucket_name)
# Remove: st.write("Files found in bucket:", s3_files)
selected_file = st.sidebar.selectbox("Choose S3 file to analyze", s3_files)

# Global Sidebar Toggles
st.sidebar.markdown("---")
show_profiling = st.sidebar.checkbox(" Show Data Profiling")
#show_ml_demo = st.sidebar.checkbox("🤖 Show ML Use Case Demo")




# ============================= TABS =============================
tab_viz, = st.tabs(["1️⃣ v"])
with tab_viz:
    def read_s3_csv_chunk(bucket, key, nrows=1000):
        s3 = boto3.client('s3')
        obj = s3.get_object(Bucket=bucket, Key=key)
        # Polars
        try:
            return pl.read_csv(obj['Body'], n_rows=nrows)
        except:
            # Fallback for Excel
           
            return pl.from_pandas(pd.read_excel(obj['Body']))

if selected_file:
    df = read_s3_csv_chunk(bucket_name, selected_file, nrows=1000)  # Load 1,000 rows for preview
    st.session_state['df'] = df
    st.subheader(" Dataset Preview")
    st.write(f"**Shape:** {df.height} rows × {df.width} columns")
    st.dataframe(df.head(10), use_container_width=True)
else:
    st.info("Please upload or select a dataset from S3 to begin.")
