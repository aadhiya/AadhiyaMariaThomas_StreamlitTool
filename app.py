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
    # Use cleaned if exists, else the original
    df = st.session_state.get('cleaned_df') or st.session_state.get('df')
    
    # Sample for plotting, keeps memory use low
    if df is not None and df.height > 1000:
        df_plot = df.head(1000)
    else:
        df_plot = df
    
    if df_plot is None or df_plot.is_empty():
        st.info("Upload or clean data to plot.")
    else:
        st.subheader("Visualizations")
        viz_type = st.radio("Choose Visualization:", ["Histogram", "Bar Chart", "Correlation Heatmap"], horizontal=True)
        st.write(f"Plotting sample of {df_plot.height} rows × {df_plot.width} columns.")

        if viz_type == "Histogram":
            advanced_mode = st.checkbox("Advanced Mode: Compare Multiple Columns", value=False)
            numeric_cols = [col for col, dtype in zip(df_plot.columns, df_plot.dtypes) if dtype in numeric_polars_types]
            good_numeric_cols = [col for col in numeric_cols if df_plot[col].drop_nulls().n_unique() > 2]
            if not advanced_mode:
                if good_numeric_cols:
                    selected_hist_col = st.selectbox("Select a numeric column to plot histogram:", good_numeric_cols, key="hist_col_select")
                    data_series = df_plot[selected_hist_col].to_pandas().dropna()
                    if len(data_series) > 1:
                        min_val, max_val = float(data_series.min()), float(data_series.max())
                        range_slider = st.slider(
                            f"Select range for {selected_hist_col}:",
                            min_value=min_val,
                            max_value=max_val,
                            value=(min_val, max_val),
                        )
                        filtered_data = data_series[(data_series >= range_slider[0]) & (data_series <= range_slider[1])]
                        fig, ax = plt.subplots()
                        ax.hist(filtered_data, bins=30, color="skyblue")
                        ax.set_xlabel(selected_hist_col)
                        ax.set_ylabel("Frequency")
                        st.pyplot(fig)
                        del filtered_data, data_series, fig, ax  # memory safety
                    else:
                        st.info("Selected column does not have sufficient unique values for histogram.")
                else:
                    st.info("No suitable numeric columns found for histogram plotting.")
            else:
                selected_cols = st.multiselect(
                    "Select numeric columns for side-by-side histograms:",
                    good_numeric_cols,
                    default=good_numeric_cols[:2] if len(good_numeric_cols) >= 2 else good_numeric_cols
                )
                if selected_cols:
                    n_cols = len(selected_cols)
                    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 4))
                    if n_cols == 1:
                        axes = [axes]
                    for ax, col in zip(axes, selected_cols):
                        data_series = df_plot[col].to_pandas().dropna()
                        ax.hist(data_series, bins=30, color="skyblue", alpha=0.8)
                        ax.set_title(f"Histogram of {col}")
                        ax.set_xlabel(col)
                        ax.set_ylabel("Frequency")
                    plt.tight_layout()
                    st.pyplot(fig)
                    del data_series, fig, axes  # memory safety
                else:
                    st.info("Please select at least one column.")

        elif viz_type == "Bar Chart":
            categorical_cols = [col for col, dtype in zip(df_plot.columns, df_plot.dtypes) if dtype == pl.Utf8]
            good_categorical_cols = [col for col in categorical_cols if df_plot[col].drop_nulls().n_unique() < 100]
            if good_categorical_cols:
                selected_cat_col = st.selectbox("Select a categorical column for bar chart:", good_categorical_cols, key="bar_cat_select")
                vc_pd = df_plot.to_pandas()[selected_cat_col].value_counts().sort_values(ascending=False)
                fig, ax = plt.subplots(figsize=(8,4))
                ax.bar(vc_pd.index.astype(str), vc_pd.values, color="mediumpurple")
                ax.set_xlabel(selected_cat_col)
                ax.set_ylabel("Counts")
                ax.set_title(f"Value Counts for {selected_cat_col}")
                plt.xticks(rotation=45, ha='right')
                st.pyplot(fig)
                del vc_pd, fig, ax  # memory safety
            else:
                st.info("No suitable categorical columns found for bar charts.")

        elif viz_type == "Correlation Heatmap":
            numeric_cols = [col for col, dtype in zip(df_plot.columns, df_plot.dtypes) if dtype in numeric_polars_types]
            if len(numeric_cols) >= 2:
                selected_corr_cols = st.multiselect(
                    "Select numeric columns for correlation matrix:",
                    numeric_cols,
                    default=numeric_cols[:5] if len(numeric_cols) >= 5 else numeric_cols
                )
                if len(selected_corr_cols) >= 2:
                    corr_data = df_plot.select(selected_corr_cols).to_pandas().corr()
                    fig, ax = plt.subplots(figsize=(1 + len(selected_corr_cols), 1 + len(selected_corr_cols)))
                    sns.heatmap(corr_data, annot=True, cmap="YlGnBu", ax=ax)
                    st.pyplot(fig)
                    del corr_data, fig, ax  # memory safety
                else:
                    st.info("Select at least two columns for correlation heatmap.")
            else:
                st.info("Not enough numeric columns for correlation heatmap.")
        
        # Final cleanup after plotting
        del df_plot
        import gc
        gc.collect()
