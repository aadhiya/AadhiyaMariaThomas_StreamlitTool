import streamlit as st
import polars as pl
import pandas as pd
import matplotlib.pyplot as plt
import io
import seaborn as sns
import numpy as np
import locale

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

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
numeric_polars_types = [
    pl.Int8, pl.Int16, pl.Int32, pl.Int64,
    pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
    pl.Float32, pl.Float64
]

df = None
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
        

        # Optionally: force conversion (skip if you want only inferred types)
      # for col in df.columns:
            # Try to coerce to float if possible (non-numeric remain unchanged)
    #       try:
    #           df = df.with_columns(
    #               pl.col(col).cast(pl.Float64, strict=False).alias(col)
    #            )
    #       except Exception:
    #            pass
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
            show_categorical = st.sidebar.checkbox("Categorical Summary", value=False)
            show_missing = st.sidebar.checkbox("Missing Data Summary", value=False)
            show_column_explorer = st.sidebar.checkbox("Interactive Column Explorer", value=False)
            show_column_explanation = st.sidebar.checkbox("Column Explanation", value=False)
            st.markdown("---")
            st.subheader("📈 Data Profiling Results") 
               
# Numeric summary

            if show_numeric:
                numeric_cols = [col for col, dtype in zip(df.columns, df.dtypes) if dtype in numeric_polars_types]
                st.write("Detected numeric columns:", numeric_cols)
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
                            st.write("Detected categorical columns:", categorical_cols)
                            if categorical_cols:
                                for col in categorical_cols:
                                    counts_df = df.select(pl.col(col).value_counts().sort( descending=True)).to_pandas()
                                    with st.expander(f"📦 Value counts for {col}"):
                                        st.dataframe(counts_df)
                            else:
                                st.info("No categorical columns found.")
# Categorical Encoding
                            st.markdown("#### 🔤 Encode Categorical Feature(s)")
                            col_to_encode = st.multiselect(
                                "Select categorical column(s) to encode (Label Encoding):",
                                categorical_cols
                            )

                            if col_to_encode:
                                st.write("**To be encoded:**", ", ".join(col_to_encode))
                                if st.button("Apply Label Encoding to selected column(s)"):
                                    for col in col_to_encode:
                                        unique_vals = df[col].unique().to_list()
                                        encoding_map = {val: i for i, val in enumerate(unique_vals)}
                                        df = df.with_columns(
                                            pl.col(col).replace(encoding_map).alias(col + "_LE")
                                        )
                                    st.success(f" Label Encoding applied to: {', '.join(col_to_encode)} (new columns: {', '.join([c + '_LE' for c in col_to_encode])})")
                                    st.dataframe(df[[*col_to_encode, *[c + '_LE' for c in col_to_encode]]].head())
# Missing Data Summary
            if show_missing:
                            st.markdown("### ❗ Missing Data Summary")
                            missing_counts = {col: int(df[col].null_count()) for col in df.columns}
                            st.table(missing_counts)
                            st.markdown("#### 🛠 Handle Missing Values")
                            cols_to_change = []
                            # Multiselect for columns - allows search and multiple selections
                            selected_cols = st.multiselect(
                                "Select column(s) to clean:",
                                df.columns,
                                placeholder="Type to search columns..."
                            )

                            # If no column is chosen, allow the special "All numeric columns" shortcut
                            apply_to_all_numeric = st.checkbox("Apply to ALL numeric columns")

                            # Show chosen columns preview
                            if selected_cols and not apply_to_all_numeric:
                                cols_to_change = selected_cols
                                st.write("**Selected Columns:**", ", ".join(cols_to_change))
                            elif apply_to_all_numeric:
                                cols_to_change = [c for c, t in zip(df.columns, df.dtypes)
                                                if "int" in str(t) or "float" in str(t)]
                                st.write("**Selected Columns:**", ", ".join(cols_to_change))
                            else:
                                st.info("Select at least one column, or use 'All numeric columns' option.")
                            method = st.radio(
                                "Choose a method:",
                                ["Fill with Mean", "Fill with Median", "Fill with Mode", "Drop Rows", "Drop Column"]
                            )

                            if st.button("Apply Missing Value Handling"):
                                if not cols_to_change:
                                    st.warning("Please select columns or choose 'All numeric columns'.")
                                else:
                                    # Perform operation
                                    if method == "Drop Column":
                                        df = df.drop(cols_to_change)
                                    elif method == "Drop Rows":
                                        df = df.drop_nulls(subset=cols_to_change)
                                    else:
                                        for c in cols_to_change:
                                            if method == "Fill with Mean":
                                                df = df.with_columns(pl.col(c).fill_null(df[c].mean()))
                                            elif method == "Fill with Median":
                                                df = df.with_columns(pl.col(c).fill_null(df[c].median()))
                                            elif method == "Fill with Mode":
                                                mode_val = df[c].drop_nulls().mode()[0]
                                                df = df.with_columns(pl.col(c).fill_null(mode_val))

                            st.success(f" Applied '{method}' to columns: {', '.join(cols_to_change)}")
                            st.dataframe(df.head())
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
show_ml_demo = st.sidebar.checkbox(" Show ML Use Case Demo")
if show_ml_demo:
    st.sidebar.markdown("---")
    st.sidebar.header("ML Use Case Demo")
    ml_use_case = st.sidebar.radio(
        "Select a Machine Learning use case:",
        [
            "Credit Default Prediction",
            "Credit Limit Estimation (Regression)"
        ]
    )

    if ml_use_case == "Credit Default Prediction":
        st.sidebar.markdown("#### Use Case: Credit Default Prediction")
        st.sidebar.write("**Goal:** Predict risk of applicant defaulting on credit using demographic and financial features.")
        st.sidebar.write("**Features Used:**")
        st.sidebar.write("""
        - Gender (`CODE_GENDER`)
        - Age (`DAYS_BIRTH`)
        - Number of Children (`CNT_CHILDREN`)
        - Income (`AMT_INCOME_TOTAL`)
        - Credit Amount (`AMT_CREDIT`)
        - Annuity (`AMT_ANNUITY`)
        - Goods Price (`AMT_GOODS_PRICE`)
        - Days Employed (`DAYS_EMPLOYED`)
        - Own Car/Realty (`FLAG_OWN_CAR`, `FLAG_OWN_REALTY`)
        - Housing Type (`NAME_HOUSING_TYPE`)
        - Organization Type (`ORGANIZATION_TYPE`)
        """)
        st.sidebar.write("**Model Used:** Logistic Regression")
        st.sidebar.write("_A logistic regression model classifies applications as 'default' or 'not default' based on provided features. You can test predictions by entering new data points._")

    elif ml_use_case == "Credit Limit Estimation (Regression)":
        st.sidebar.markdown("#### Use Case: Credit Limit Estimation")
        st.sidebar.write("**Goal:** Estimate expected credit limit (`AMT_CREDIT`) for an applicant based on profile features.")
        st.sidebar.write("**Features Used:**")
        st.sidebar.write("""
        - Income (`AMT_INCOME_TOTAL`)
        - Age (`DAYS_BIRTH`)
        - Number of Children (`CNT_CHILDREN`)
        - Employment Duration (`DAYS_EMPLOYED`)
        - Housing/Organization Type
        """)
        st.sidebar.write("**Model Used:** Linear Regression")
        st.sidebar.write("_A linear regression model predicts the amount of credit that can be granted. You can input sample data to get a prediction._")

else:
    st.info("👆 Please upload a CSV file.")
# VISUALIZATIONS SECTION =========
st.markdown("---")
st.subheader("Visualizations")

if df is not None:
    viz_type = st.radio(
        "Choose Visualization Type:",
        ("Histogram", "Bar Chart", "Correlation Heatmap"),
        horizontal=True
    )

# --- Histogram (Single & Advanced) ---
    if viz_type == "Histogram":
        advanced_mode = st.checkbox("Advanced Mode: Compare Multiple Columns", value=False)
        numeric_cols = [col for col, dtype in zip(df.columns, df.dtypes) if dtype in numeric_polars_types]
        good_numeric_cols = [col for col in numeric_cols if df[col].drop_nulls().n_unique() > 2]
        if not advanced_mode:
            if good_numeric_cols:
                selected_hist_col = st.selectbox(
                    "Select a numeric column to plot histogram:",
                    good_numeric_cols,
                    key="hist_col_select"
                )
                data_series = df[selected_hist_col].to_pandas().dropna()
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
                else:
                    st.info("Selected column does not have sufficient unique values for histogram.")
            else:
                st.info("No suitable numeric columns found in your dataset for histogram plotting.")
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
                    data_series = df[col].to_pandas().dropna()
                    ax.hist(data_series, bins=30, color="skyblue", alpha=0.8)
                    ax.set_title(f"Histogram of {col}")
                    ax.set_xlabel(col)
                    ax.set_ylabel("Frequency")
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.info("Please select at least one column.")

# --- Bar Chart (Categorical) ---
    elif viz_type == "Bar Chart":
        categorical_cols = [col for col, dtype in zip(df.columns, df.dtypes) if dtype == pl.Utf8]
        good_categorical_cols = [col for col in categorical_cols if df[col].drop_nulls().n_unique() < 100]
        if good_categorical_cols:
            selected_cat_col = st.selectbox(
                "Select a categorical column for bar chart:",
                good_categorical_cols,
                key="bar_cat_select"
            )
            vc_pd = df.to_pandas()[selected_cat_col].value_counts().sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=(8,4))
            ax.bar(vc_pd.index.astype(str), vc_pd.values, color="mediumpurple")
            ax.set_xlabel(selected_cat_col)
            ax.set_ylabel("Counts")
            ax.set_title(f"Value Counts for {selected_cat_col}")
            plt.xticks(rotation=45, ha='right')
            st.pyplot(fig)
        else:
            st.info("No suitable categorical columns found for bar charts.")

    # --- Correlation Heatmap ---
    elif viz_type == "Correlation Heatmap":
        
        numeric_cols = [col for col, dtype in zip(df.columns, df.dtypes) if dtype in numeric_polars_types]
        if len(numeric_cols) >= 2:
            selected_corr_cols = st.multiselect(
                "Select numeric columns for correlation matrix:",
                numeric_cols,
                default=numeric_cols[:5] if len(numeric_cols) >= 5 else numeric_cols
            )
            if len(selected_corr_cols) >= 2:
                corr_data = df.select(selected_corr_cols).to_pandas().corr()
                fig, ax = plt.subplots(figsize=(1 + len(selected_corr_cols), 1 + len(selected_corr_cols)))
                sns.heatmap(corr_data, annot=True, cmap="YlGnBu", ax=ax)
                st.pyplot(fig)
            else:
                st.info("Select at least two columns for correlation heatmap.")
        else:
            st.info("Not enough numeric columns for correlation heatmap.")        
else:
    st.info("Please upload a dataset first to see visualizations.")

# ===== Download Cleaned File =====
st.markdown("---")
st.subheader(" Export Cleaned Data")
if df is not None:
    # Use BytesIO, to support encoding data as well
    csv_buffer = io.BytesIO()
    df.write_csv(csv_buffer)
    csv_buffer.seek(0)  # Rewind to start

    st.download_button(
        label="Download Cleaned CSV",
        data=csv_buffer,
        file_name="cleaned_data.csv",
        mime="text/csv"
    )
else:
    st.info("Please upload and clean data first to enable export.")
# ML Use Case Demo
if show_ml_demo:
    st.markdown("---")
    st.header("Machine Learning Demo")
    st.subheader(f"Selected Use Case: {ml_use_case}")
    if ml_use_case == "Credit Default Prediction":
    # --- Step 1: Prepare Data & Check Cleanliness ---
        st.subheader("Step 1: Prepare Data Quality Check")

        required_columns = [
            "TARGET", "CODE_GENDER", "DAYS_BIRTH", "CNT_CHILDREN", "AMT_INCOME_TOTAL",
            "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE", "DAYS_EMPLOYED",
            "FLAG_OWN_CAR", "FLAG_OWN_REALTY", "NAME_HOUSING_TYPE", "ORGANIZATION_TYPE"
        ]
        missing_cols = [col for col in required_columns if col not in df.columns]
        err_msgs = []
        if missing_cols:
            err_msgs.append(f"Missing required columns: {', '.join(missing_cols)}.\nUse 'Interactive Column Explorer' to check columns and 'Column Explanation' for details.")
        else:
            pd_df = df.to_pandas()
            missing_vals = pd_df[required_columns].isnull().sum()
            num_missing = missing_vals[missing_vals > 0]
            if not num_missing.empty:
                st.warning("The following features have missing values:")
                st.write(num_missing)
                st.markdown("""
                 To fix missing **numeric** columns, use 'Missing Data Summary' and fill with **mean/median** (recommended).
                 To fix missing **categorical** columns, fill with **mode** or drop rows/columns as needed using sidebar tools.
                """)
            numeric_cols = [
                "DAYS_BIRTH", "CNT_CHILDREN", "AMT_INCOME_TOTAL",
                "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE", "DAYS_EMPLOYED"
            ]
            for col in numeric_cols:
                if not pd.api.types.is_numeric_dtype(pd_df[col]):
                    err_msgs.append(f" Column '{col}' is not numeric. Use 'Interactive Column Explorer' and 'Encode Categorical Feature(s)' or fix data type in sidebar.")
            if pd_df["TARGET"].isnull().any():
                err_msgs.append(" Target column `TARGET` has missing values. Use sidebar cleaning tools to drop or fill.")
            if len(err_msgs) > 0 or not num_missing.empty:
                st.error("Data is NOT ready for ML training.")
                for msg in err_msgs:
                    st.markdown(msg)
                st.info("Please use the data cleaning options in the sidebar, then export and reload the cleaned data.")
                st.stop()
            else:
                st.success(" Data passes all checks and is ready for ML training.")
                # --- Continue with model training/prediction code here ---
            # --- Step 2: Train Model ---
            
                st.subheader("Step 2: Train Model")
                selected_model = st.selectbox("Select Model:", ["Logistic Regression"])
                categorical_cols = ["CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY", "NAME_HOUSING_TYPE", "ORGANIZATION_TYPE"]
                if st.button("Train Model"):
                    # Prepare features and target
                    features = [
                        "CODE_GENDER", "DAYS_BIRTH", "CNT_CHILDREN", "AMT_INCOME_TOTAL",
                        "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE", "DAYS_EMPLOYED",
                        "FLAG_OWN_CAR", "FLAG_OWN_REALTY", "NAME_HOUSING_TYPE", "ORGANIZATION_TYPE"
                    ]
                    st.session_state['features'] = features
                    X = pd_df[features].copy()
                    y = pd_df["TARGET"].astype(int)

                    # Fill missing values
                    for col in X.select_dtypes(include=[np.number]).columns:
                         X[col] = X[col].fillna(X[col].median())
                    for col in X.select_dtypes(include='object').columns:
                        X[col] = X[col].fillna(X[col].mode()[0])

                    # Label encode categorical columns
                    encoders = {}
                    
                    for col in categorical_cols:
                        le = LabelEncoder()
                        X[col] = le.fit_transform(X[col].astype(str))
                        encoders[col] = le

                    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

                    model = LogisticRegression(max_iter=500)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    acc = accuracy_score(y_test, y_pred)
                    # SAVE model and encoders to st.session_state:
                    st.session_state['model'] = model
                    st.session_state['encoders'] = encoders
                    st.session_state['X'] = X  # for column medians in prediction
                    st.success(f"Model trained! Accuracy on test data: {acc:.3f}")

                    cm = confusion_matrix(y_test, y_pred)
                    st.write("Confusion Matrix:")
                    st.dataframe(pd.DataFrame(cm, columns=['Pred 0', 'Pred 1'], index=['Actual 0', 'Actual 1']))
                    st.write("Classification Report:")
                    st.text(classification_report(y_test, y_pred, zero_division=0))

                    st.write("Feature Importances (coefficients):")
                    st.dataframe(pd.DataFrame({"Feature": features, "Importance": model.coef_[0]}).sort_values(by="Importance", ascending=False))
                    
                # Step 3: Prediction Input
                st.subheader("Step 3: Predict Default Risk on New Applicant")

                model = st.session_state.get('model')
                encoders = st.session_state.get('encoders')
                X = st.session_state.get('X')
                

                # Only show prediction step if model exists
                if model is not None and encoders is not None and X is not None:
                    user_input = {}
                    # Categorical inputs
                    for col in categorical_cols:
                        options = list(encoders[col].classes_)
                        user_val = st.selectbox(col, options)
                        user_input[col] = encoders[col].transform([user_val])[0]

                    # Numeric inputs (as before)
                    numeric_cols = list(X.select_dtypes(include=[np.number]).columns.difference(categorical_cols))
                    for col in numeric_cols:
                        median_val = float(X[col].median())
                        user_input[col] = st.number_input(col, value=median_val)

                    if st.button("Predict"):
                        
                        try:
                            # Retrieve feature list saved during training
                            features = st.session_state.get('features')
                            
                            input_df = pd.DataFrame([user_input])

                            # Reorder input_df columns to exactly match training features
                            input_df = input_df[features]

                            st.write("Input dataframe for prediction:", input_df)
                            
                            prob = model.predict_proba(input_df)[0][1]
                            pred = model.predict(input_df)
                            st.info(f"Predicted probability of default: {prob:.2%}")
                            st.write(f"Prediction: {'Default Risk' if pred == 1 else 'Low Risk'}")
                        except Exception as e:
                            st.error(f"Prediction error: {e}")


                else:
                    st.info("Please train the model first.")



    elif ml_use_case == "Credit Limit Estimation (Regression)":
        # --- Step 1: Prepare Data & Check Cleanliness ---
        st.subheader("Step 1: Prepare Data Quality Check")

        required_columns = [
        "AMT_CREDIT", "AMT_INCOME_TOTAL", "DAYS_BIRTH", "CNT_CHILDREN",
        "DAYS_EMPLOYED", "NAME_HOUSING_TYPE", "ORGANIZATION_TYPE"
    ]
    missing_cols = [col for col in required_columns if col not in df.columns]
    err_msgs = []
    if missing_cols:
        err_msgs.append(f"Missing required columns: {', '.join(missing_cols)}.\nUse 'Interactive Column Explorer' to check columns and 'Column Explanation' for details.")
    else:
        pd_df = df.to_pandas()
        missing_vals = pd_df[required_columns].isnull().sum()
        num_missing = missing_vals[missing_vals > 0]
        if not num_missing.empty:
            st.warning("The following features have missing values:")
            st.write(num_missing)
            st.markdown("""
                To fix missing **numeric** columns, use 'Missing Data Summary' and fill with **mean/median** (recommended).
                To fix missing **categorical** columns, fill with **mode** or drop rows/columns as needed using sidebar tools.
            """)

        numeric_cols = [
            "AMT_CREDIT", "AMT_INCOME_TOTAL", "DAYS_BIRTH", "CNT_CHILDREN", "DAYS_EMPLOYED"
        ]
        for col in numeric_cols:
            if not pd.api.types.is_numeric_dtype(pd_df[col]):
                err_msgs.append(f"Column '{col}' is not numeric. Use 'Interactive Column Explorer' and 'Encode Categorical Feature(s)' or fix data type in sidebar.")

        if len(err_msgs) > 0 or not num_missing.empty:
            st.error("Data is NOT ready for ML training.")
            for msg in err_msgs:
                st.markdown(msg)
            st.info("Please use the data cleaning options in the sidebar, then export and reload the cleaned data.")
            st.stop()
        else:
            st.success("Data passes all checks and is ready for ML training.")

            # --- Step 2: Train Model ---
            st.subheader("Step 2: Train Model")

            categorical_cols = ["CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY", "NAME_HOUSING_TYPE", "ORGANIZATION_TYPE"]

            features = [
                "CODE_GENDER", "DAYS_BIRTH", "CNT_CHILDREN", "AMT_INCOME_TOTAL",
                "AMT_ANNUITY", "AMT_GOODS_PRICE", "DAYS_EMPLOYED",
                "FLAG_OWN_CAR", "FLAG_OWN_REALTY", "NAME_HOUSING_TYPE", "ORGANIZATION_TYPE"
            ]
            st.session_state['features'] = features

            selected_model = st.selectbox("Select Model:", ["Linear Regression"])

            if st.button("Train Model"):
                X = pd_df[features].copy()
                y = pd_df["AMT_CREDIT"].astype(float)

                # Fill missing values
                for col in X.select_dtypes(include=[np.number]).columns:
                    X[col] = X[col].fillna(X[col].median())
                for col in X.select_dtypes(include='object').columns:
                    X[col] = X[col].fillna(X[col].mode()[0])

                # Label encode categorical columns
                encoders = {}
                for col in categorical_cols:
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))
                    encoders[col] = le

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                from sklearn.linear_model import LinearRegression
                model = LinearRegression()
                model.fit(X_train, y_train)

                y_pred = model.predict(X_test)
                from sklearn.metrics import mean_squared_error, r2_score
                locale.setlocale(locale.LC_ALL, '') 
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                rmse = mse ** 0.5
                # Save model and encoders for prediction
                st.session_state['model'] = model
                st.session_state['encoders'] = encoders
                st.session_state['X'] = X

                st.success(f"Model trained successfully!")
                st.write(f"R² score (Goodness of fit): **{r2 * 100:.2f}%**")
                st.write(f"Root Mean Squared Error (RMSE): **₹{rmse:,.0f}** (average prediction error)")

            # Step 3: Prediction Input
            st.subheader("Step 3: Predict Credit Limit for New Applicant")

            model = st.session_state.get('model')
            encoders = st.session_state.get('encoders')
            X = st.session_state.get('X')

            if model is not None and encoders is not None and X is not None:
                user_input = {}
                for col in categorical_cols:
                    options = list(encoders[col].classes_)
                    user_val = st.selectbox(col, options)
                    user_input[col] = encoders[col].transform([user_val])[0]

                numeric_cols = list(X.select_dtypes(include=[np.number]).columns.difference(categorical_cols))
                for col in numeric_cols:
                    median_val = float(X[col].median())
                    user_input[col] = st.number_input(col, value=median_val)

                if st.button("Predict"):
                    try:
                        features = st.session_state.get('features')
                        input_df = pd.DataFrame([user_input])
                        input_df = input_df[features]
                        pred = model.predict(input_df)[0]

                        predicted_limit = pred
                        st.info(f"Predicted Credit Limit for this applicant: **₹{predicted_limit:,.0f}**")
                    except Exception as e:
                        st.error(f"Prediction error: {e}")
            else:
                st.info("Please train the model first.")