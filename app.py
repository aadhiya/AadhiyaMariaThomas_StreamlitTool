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
file_key = "application_data_75_percent.csv" 

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


def s3_select_csv_head(bucket, key, nrows=1):
    s3 = boto3.client('s3')
    sql_exp = f"SELECT * FROM S3Object LIMIT {nrows}"
    response = s3.select_object_content(
        Bucket=bucket,
        Key=key,
        ExpressionType='SQL',
        Expression=sql_exp,
        InputSerialization={'CSV': {"FileHeaderInfo": "USE"}, "CompressionType": "NONE"},
        OutputSerialization={'CSV': {}},
    )
    rows = ""
    for event in response['Payload']:
        if 'Records' in event:
            rows += event['Records']['Payload'].decode()
    if len(rows.strip()) == 0:
        return None
    return pd.read_csv(io.StringIO(rows))

# ============================= TABS =============================
tab_viz, = st.tabs(["1️⃣ v"])
with tab_viz:
    st.subheader("Preview first row using AWS S3 Select")

    if selected_file and selected_file.lower().endswith('.csv'):
        try:
            df_sample = s3_select_csv_head(bucket_name, selected_file, nrows=1)
            if df_sample is not None and not df_sample.empty:
                st.write("First row from your S3 file (via S3 Select):")
                st.dataframe(df_sample)
            else:
                st.warning("S3 Select query succeeded but no data was returned (empty row).")
        except Exception as e:
            st.error(f"S3 Select failed: {e}")
            st.info("Check if your file is a plain CSV (not zipped or encrypted), and that your IAM role has S3 Select permissions.")
    elif selected_file:
        st.warning("S3 Select is only supported for plain CSV files.")
    else:
        st.info("Please select a file from S3.")
