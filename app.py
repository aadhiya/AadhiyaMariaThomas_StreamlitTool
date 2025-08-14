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

else:
    st.info("👆 Please upload a CSV or Excel file.")

#Descriptive stats for numeric columns
def numeric_profile(df):
    numeric_cols = [col for col, dtype in zip(df.columns, df.dtypes) if dtype in [pl.Int64, pl.Float64, pl.Float32, pl.Int32]]
    if not numeric_cols:
        return None

    stats_df = df.select([
        pl.col(c).mean().alias(f'{c}_mean') for c in numeric_cols
    ] + [
        pl.col(c).median().alias(f'{c}_median') for c in numeric_cols
    ] + [
        pl.col(c).std().alias(f'{c}_std') for c in numeric_cols
    ] + [
        pl.col(c).min().alias(f'{c}_min') for c in numeric_cols
    ] + [
        pl.col(c).max().alias(f'{c}_max') for c in numeric_cols
    ] + [
        pl.col(c).is_null().sum().alias(f'{c}_missing') for c in numeric_cols
    ])

    return stats_df.to_pandas()

#Provide counts/distributions for categorical columns
def categorical_profile(df):
    categorical_cols = [col for col, dtype in zip(df.columns, df.dtypes) if dtype == pl.Utf8]
    summary = {}
    for col in categorical_cols:
        count_df = df.select([
            pl.col(col).value_counts().sort(descending=True)
        ]).to_pandas()
        summary[col] = count_df
    return summary

#missing data summary
def missing_data_summary(df):
    missing_counts = {col: df[col].null_count() for col in df.columns}
    return missing_counts


# Assuming df is a Polars DataFrame loaded after upload

st.subheader("Data Profiling")

# Numeric summary
numeric_stats = numeric_profile(df)
if numeric_stats is not None:
    st.write("### Numeric Columns Summary")
    st.dataframe(numeric_stats)

# Categorical summary
cat_summary = categorical_profile(df)
st.write("### Categorical Columns Summary")
for col, counts in cat_summary.items():
    with st.expander(f"Value counts for {col}"):
        st.dataframe(counts)

# Missing data
missing_summary = missing_data_summary(df)
st.write("### Missing Data Summary")
st.table(missing_summary)

# Interactive column explorer
col_to_explore = st.selectbox("Select a column to explore:", df.columns)

if col_to_explore:
    col_dtype = df[col_to_explore].dtype
    st.write(f"Exploring column `{col_to_explore}` of type `{col_dtype}`")

    unique_vals = df[col_to_explore].unique().to_list()
    st.write(f"Unique values count: {len(unique_vals)}")

    if col_dtype in [pl.Int64, pl.Float64, pl.Float32, pl.Int32]:
        data_series = df[col_to_explore].to_pandas()
        st.write(f"Descriptive stats:")
        st.write(data_series.describe())

        fig, ax = plt.subplots()
        ax.hist(data_series.dropna(), bins=30, color='skyblue')
        ax.set_title(f"Histogram of {col_to_explore}")
        st.pyplot(fig)

    elif col_dtype == pl.Utf8:
        counts = df.select(pl.col(col_to_explore).value_counts()).to_pandas()
        st.write(counts)
        fig, ax = plt.subplots()
        ax.bar(counts[col_to_explore], counts['counts'])
        ax.set_xticklabels(counts[col_to_explore], rotation=45, ha='right')
        ax.set_title(f"Value Counts for {col_to_explore}")
        st.pyplot(fig)
