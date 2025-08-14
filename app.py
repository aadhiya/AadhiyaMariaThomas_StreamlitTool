import streamlit as st
import polars as pl
import pandas as pd
import matplotlib.pyplot as plt
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

# Step 1: File uploader widget (this creates the variable)
uploaded_file = st.file_uploader(
    "Upload your dataset (CSV or Excel)",
    type=["csv", "xlsx"],
    help="Select a .csv or .xlsx file from your computer"
)

# Step 2: Only process if file exists
if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            try:
                df = pl.read_csv(
                    uploaded_file,
                    infer_schema_length=10000,
                    ignore_errors=True
                )
            except Exception:
                uploaded_file.seek(0)
                df_pd = pd.read_csv(uploaded_file)
                df = pl.from_pandas(df_pd)
        else:
            df = pl.from_pandas(pd.read_excel(uploaded_file))

        st.write(f"**Shape:** {df.height} rows × {df.width} columns")
        st.dataframe(df.head(10))

    except Exception as e:
        st.warning(f"Warning: Could not load file. Details: {e}")




# Assuming df is a Polars DataFrame loaded after upload

#st.subheader("Data Profiling")
# ========= SIDEBAR =========
st.sidebar.header("Options")
show_profiling = st.sidebar.checkbox("🔍 Show Data Profiling")
if show_profiling:
        try:
            st.sidebar.subheader("📊 Profiling Sections")
            show_numeric = st.sidebar.checkbox("Numeric Summary", value=True)
            show_categorical = st.sidebar.checkbox("Categorical Summary", value=True)
            show_missing = st.sidebar.checkbox("Missing Data Summary", value=True)
            show_column_explorer = st.sidebar.checkbox("Interactive Column Explorer", value=True)

            st.markdown("---")
            st.subheader("📈 Data Profiling Results") 
               
# Numeric summary

            if show_numeric:
                numeric_cols = [col for col, dtype in zip(df.columns, df.dtypes) 
                                if dtype in [pl.Int64, pl.Float64, pl.Float32, pl.Int32]]
                if numeric_cols:
                    st.markdown("### 📌 Numeric Columns Summary")
                    stats_df = df.select(
                        [pl.col(c).mean().alias(f"{c}_mean") for c in numeric_cols] +
                        [pl.col(c).median().alias(f"{c}_median") for c in numeric_cols] +
                        [pl.col(c).std().alias(f"{c}_std") for c in numeric_cols] +
                        [pl.col(c).min().alias(f"{c}_min") for c in numeric_cols] +
                        [pl.col(c).max().alias(f"{c}_max") for c in numeric_cols] +
                        [pl.col(c).is_null().sum().alias(f"{c}_missing") for c in numeric_cols]
                    ).to_pandas()
                    st.dataframe(stats_df, use_container_width=True)
                else:
                    st.info("No numeric columns found.")

# Categorical Summary
                if show_categorical:            
                                st.markdown("### 🏷 Categorical Columns Summary")
                                categorical_cols = [col for col, dtype in zip(df.columns, df.dtypes) if dtype == pl.Utf8]
                                if categorical_cols:
                                    for col in categorical_cols:
                                        counts_df = df.select(pl.col(col).value_counts().sort( descending=True)).to_pandas()
                                        with st.expander(f"📦 Value counts for {col}"):
                                            st.dataframe(counts_df)
                                else:
                                    st.info("No categorical columns found.")
 # Missing Data Summary
                if show_missing:
                                st.markdown("### ❗ Missing Data Summary")
                                missing_counts = {col: int(df[col].null_count()) for col in df.columns}
                                st.table(missing_counts)

                # Interactive Column Explorer
                if show_column_explorer:
                                st.markdown("### 🔍 Interactive Column Explorer")
                                col_to_explore = st.selectbox("Select a column to explore:", df.columns)
                                col_dtype = df[col_to_explore].dtype

                                if col_dtype in [pl.Int64, pl.Float64, pl.Float32, pl.Int32]:
                                    series = df[col_to_explore].to_pandas()
                                    fig, ax = plt.subplots()
                                    ax.hist(series.dropna(), bins=30, color="skyblue")
                                    ax.set_title(f"Histogram of {col_to_explore}")
                                    st.pyplot(fig)
                                elif col_dtype == pl.Utf8:
                                    counts = df.select(pl.col(col_to_explore).value_counts().sort("counts", descending=True)).to_pandas()
                                    fig, ax = plt.subplots()
                                    ax.bar(counts[col_to_explore], counts["counts"])
                                    plt.xticks(rotation=45, ha='right')
                                    st.pyplot(fig)

        except Exception as e:
                    st.error(f"Error during data profiling: {e}")


else:
    st.info("👆 Please upload a CSV file.")