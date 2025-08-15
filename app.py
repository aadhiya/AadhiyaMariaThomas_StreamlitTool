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
show_profiling = st.sidebar.checkbox(" Show Data Profiling")
if show_profiling:
        try:
            st.sidebar.subheader(" Profiling Sections")
            show_numeric = st.sidebar.checkbox("Numeric Summary", value=True)
            show_categorical = st.sidebar.checkbox("Categorical Summary", value=True)
            show_missing = st.sidebar.checkbox("Missing Data Summary", value=True)
            show_column_explorer = st.sidebar.checkbox("Interactive Column Explorer", value=True)
            show_column_explanation = st.sidebar.checkbox("Column Explanation", value=False)
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
                            st.markdown("#### 🛠 Handle Missing Values")

                            # Let the user choose a column or "All numeric columns"
                            target_col = st.sidebar.multiselect(
                            "Type to search and select column(s) to clean:",
                            ["All numeric columns"] + list(df.columns),
                            max_selections=2  # if you want single-select behavior
                            )
                            if target_col:
                                target_col = target_col[0]  # if you want just one column
                            else:
                                target_col = "All numeric columns"


                            method = st.radio(
                                "Choose a method:",
                                ["Fill with Mean", "Fill with Median", "Fill with Mode", "Drop Rows", "Drop Column"]
                            )

                            if st.button("Apply Missing Value Handling"):
                                if method == "Drop Column":
                                    if target_col == "All numeric columns":
                                        st.warning("Please select a specific column to drop.")
                                    else:
                                        df = df.drop(target_col)
                                elif method == "Drop Rows":
                                    if target_col == "All numeric columns":
                                        df = df.drop_nulls()
                                    else:
                                        df = df.drop_nulls(subset=[target_col])
                                else:
                                    import statistics
                                    if target_col == "All numeric columns":
                                        num_cols = [c for c,t in zip(df.columns, df.dtypes) if "int" in str(t) or "float" in str(t)]
                                        for c in num_cols:
                                            if method == "Fill with Mean":
                                                df = df.with_columns(pl.col(c).fill_null(df[c].mean()))
                                            elif method == "Fill with Median":
                                                df = df.with_columns(pl.col(c).fill_null(df[c].median()))
                                            elif method == "Fill with Mode":
                                                mode_val = df[c].drop_nulls().mode()[0]
                                                df = df.with_columns(pl.col(c).fill_null(mode_val))
                                    else:
                                        if method == "Fill with Mean":
                                            df = df.with_columns(pl.col(target_col).fill_null(df[target_col].mean()))
                                        elif method == "Fill with Median":
                                            df = df.with_columns(pl.col(target_col).fill_null(df[target_col].median()))
                                        elif method == "Fill with Mode":
                                            mode_val = df[target_col].drop_nulls().mode()[0]
                                            df = df.with_columns(pl.col(target_col).fill_null(mode_val))

                                st.success(" Missing value handling applied!")
                                # Optionally show updated preview
                                st.dataframe(df.head())

# ===== Duplicate Handling =====
            st.markdown("###  Duplicate Records Check")

            # Count duplicates
            df_pandas = df.to_pandas()
            duplicate_count = df_pandas.duplicated().sum()

            st.write(f"**Found {duplicate_count} duplicate rows.**")

            if duplicate_count > 0:
                if st.button("Remove Duplicate Rows"):
                    before_rows = df.height
                    df = df.unique()
                    after_rows = df.height
                    removed_count = before_rows - after_rows
                    st.success(f"Removed {removed_count} duplicates. New row count: {after_rows}")
            else:
                st.info("No duplicate rows found.")

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
#column explanation
            if show_column_explanation:
                st.sidebar.markdown("---")
                st.sidebar.write("Optional: Upload a column description CSV")
                col_desc_file = st.sidebar.file_uploader(
                    "Upload Column Descriptions", 
                    type=["csv"],
                    key="col_desc"
                )

                st.markdown("###  Column Explanation")
                if col_desc_file is not None:
                    try:
                        try:
                            desc_df = pd.read_csv(col_desc_file, encoding='utf-8')
                        except UnicodeDecodeError:
                            desc_df = pd.read_csv(col_desc_file, encoding='latin1')

                        # Normalize column names (strip and lowercase)
                        desc_df.columns = [c.strip() for c in desc_df.columns]

                        # If file uses "Row" instead of "Column", rename it
                        if 'Row' in desc_df.columns and 'Column' not in desc_df.columns:
                            desc_df.rename(columns={'Row': 'Column'}, inplace=True)

                        required_cols = {"Column", "Description"}
                        if not required_cols.issubset(desc_df.columns):
                            st.error(f"Description file must contain columns: {required_cols}")
                        else:
                            desc_df = desc_df[desc_df["Column"].isin(df.columns)]
                            st.dataframe(desc_df, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error reading column description file: {e}")
                else:
                    # Fallback to auto-generated descriptions
                    explanations = []
                    for col, dtype in zip(df.columns, df.dtypes):
                        dtype_str = str(dtype)
                        sample_val = df[col].drop_nulls().head(1).to_list()
                        sample_text = f" e.g., '{sample_val[0]}'" if sample_val else ""
                        missing_count = int(df[col].null_count())

                        if "int" in dtype_str or "float" in dtype_str:
                            explanation = f"'{col}' is a numeric column containing numbers{sample_text} with {missing_count} missing values."
                        elif "str" in dtype_str or "Utf8" in dtype_str:
                            explanation = f"'{col}' is a text/categorical column{sample_text} with {missing_count} missing values."
                        else:
                            explanation = f"'{col}' is of type {dtype_str}{sample_text} with {missing_count} missing values."

                        explanations.append({"Column": col, "Description": explanation})

                    st.dataframe(pd.DataFrame(explanations))
        except Exception as e:
                    st.error(f"Error during data profiling: {e}")


else:
    st.info("👆 Please upload a CSV file.")