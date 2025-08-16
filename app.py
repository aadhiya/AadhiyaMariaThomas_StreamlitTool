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
tab_upload, tab_clean, tab_profile, tab_ml, tab_viz = st.tabs([
    "1️⃣ Upload & Preview",
    "2️⃣ Data Cleaning",
    "3️⃣ Data Profiling",
    "4️⃣ Machine Learning",
    "5️⃣ Visualizations"
])

# ---------------- TAB 1: Upload ----------------
with tab_upload:
    def read_s3_csv_chunk(bucket, key, nrows=1000):
        s3 = boto3.client('s3')
        obj = s3.get_object(Bucket=bucket, Key=key)
        try:
            return pl.read_csv(obj['Body'], n_rows=nrows)
        except:
            return pl.from_pandas(pd.read_excel(obj['Body']))

    if selected_file:
        # Read only 1000 rows for preview - DO NOT save globally
        df_preview = read_s3_csv_chunk(bucket_name, selected_file, nrows=1000)
        st.subheader(" Dataset Preview")
        st.write(f"**Shape:** {df_preview.height} rows × {df_preview.width} columns")
        st.dataframe(df_preview.head(10), use_container_width=True)

        # Explicitly delete after use to free memory
        del df_preview
        import gc
        gc.collect()
    else:
        st.info("Please upload or select a dataset from S3 to begin.")


# ---------------- TAB 2: Cleaning ----------------
with tab_clean:
    if selected_file:
        # Just load the data fresh before each cleaning action (do not use session state)
        def read_s3_csv_all(bucket, key):
            s3 = boto3.client('s3')
            obj = s3.get_object(Bucket=bucket, Key=key)
            try:
                return pl.read_csv(obj['Body'])
            except:
                return pl.from_pandas(pd.read_excel(obj['Body']))
        
        df = read_s3_csv_all(bucket_name, selected_file)

        st.subheader(" Data Cleaning")

        st.markdown("### Missing Data Summary")
        missing_counts = {col: int(df[col].null_count()) for col in df.columns}
        st.table(missing_counts)

        numeric_cols = [c for c, t in zip(df.columns, df.dtypes) if t in numeric_polars_types]
        selected_cols = st.multiselect("Select column(s) to clean:", df.columns)
        apply_all = st.checkbox("Apply to ALL numeric columns")
        if apply_all:
            selected_cols = numeric_cols

        method = st.radio("Choose method:", [
            "Fill with Mean", "Fill with Median", "Fill with Mode", "Drop Rows", "Drop Columns"
        ])

        st.write("Columns before cleaning:", df.columns)

        # Only operate on a local cleaned_df until download
        cleaned_df = df.clone()

        if st.button("Apply Missing Value Handling"):
            if not selected_cols:
                st.warning("Select columns first.")
            else:
                actions = []
                for c in selected_cols:
                    if method == "Fill with Mean":
                        cleaned_df = cleaned_df.with_columns(pl.col(c).fill_null(cleaned_df[c].mean()))
                        actions.append(f"Filled missing values in column '{c}' with mean")
                    elif method == "Fill with Median":
                        cleaned_df = cleaned_df.with_columns(pl.col(c).fill_null(cleaned_df[c].median()))
                        actions.append(f"Filled missing values in column '{c}' with median")
                    elif method == "Fill with Mode":
                        mode_val = cleaned_df[c].drop_nulls().mode()[0]
                        cleaned_df = cleaned_df.with_columns(pl.col(c).fill_null(mode_val))
                        actions.append(f"Filled missing values in column '{c}' with mode")
                    elif method == "Drop Rows":
                        before_rows = cleaned_df.height
                        cleaned_df = cleaned_df.drop_nulls(subset=selected_cols)
                        after_rows = cleaned_df.height
                        actions.append(f"Dropped {before_rows - after_rows} rows with missing data in columns: {', '.join(selected_cols)}")
                        break
                    elif method == "Drop Columns":
                        cleaned_df = cleaned_df.drop(selected_cols)
                        actions.append(f"Dropped columns: {', '.join(selected_cols)}")
                        break
                st.write("Columns after cleaning:", cleaned_df.columns)
                for action in actions:
                    st.info(action)

                # Handle duplicate removal
                duplicates = cleaned_df.to_pandas().duplicated().sum()
                st.write(f"Found **{duplicates}** duplicate rows.")
                if duplicates > 0 and st.button("Remove Duplicates"):
                    cleaned_df_pd = cleaned_df.to_pandas().drop_duplicates()
                    # Convert back to polars only if you want further cleaning
                    cleaned_df = pl.from_pandas(cleaned_df_pd)
                    st.success("Duplicates removed.")

                # Export cleaned data for download immediately after cleaning
                csv_str = cleaned_df.to_pandas().to_csv(index=False)
                csv_bytes = csv_str.encode('utf-8')
                st.download_button(
                    label=" Download Cleaned CSV",
                    data=csv_bytes,
                    file_name="cleaned_data.csv",
                    mime="text/csv"
                )

                # Free up any large variable after download
                del df, cleaned_df, csv_bytes, csv_str
                import gc
                gc.collect()
        else:
            st.info("Use the cleaning options above, then press 'Apply Missing Value Handling' to enable download.")
    else:
        st.info("Upload or select a file from S3 to use the cleaning tool.")



# ---------------- TAB 3: Profiling ----------------
with tab_profile:
    df = st.session_state.get('cleaned_df') or st.session_state.get('df')
    if df is None or df.is_empty():
        st.info("Upload to view profiling.")
    elif not show_profiling:
        st.info("Enable profiling from sidebar.")
    else:
        st.subheader("📈 Data Profiling Summary")
        st.write(f"Profiling sample: {df.height} rows loaded from S3 (chunked)")
        
        # Numeric summary
        num_cols = [c for c, t in zip(df.columns, df.dtypes) if t in numeric_polars_types]
        if num_cols:
            st.markdown("#### Numeric Summary")
            stats_df = df.select(
                [pl.col(c).mean().alias(f"{c}_mean") for c in num_cols] +
                [pl.col(c).median().alias(f"{c}_median") for c in num_cols] +
                [pl.col(c).std().alias(f"{c}_std") for c in num_cols]
            ).to_pandas()
            st.dataframe(stats_df)
        else:
            st.warning("No numeric columns found.")
        
        # Categorical summary
        cat_cols = [c for c, t in zip(df.columns, df.dtypes) if t == pl.Utf8]
        if cat_cols:
            st.markdown("#### Categorical Summary")
            for col in cat_cols:
                counts = df.select(pl.col(col).value_counts().sort(descending=True)).to_pandas()
                with st.expander(f"Value Counts: {col}"):
                    st.dataframe(counts)
        
        # --- Automated ydata-profiling (with optimization) ---
        st.markdown("---")
        st.markdown("#### Automated Data Profile Report (ydata-profiling)")
        pd_df = df.to_pandas()
        row_count = len(pd_df)

        # 1. Warning for large datasets and ask for sampling
        max_sample = 1000
        sample = False
        if row_count > max_sample:
            sample = st.checkbox(
                f"⚡ Your dataset has {row_count} rows. Tick to profile a random sample of {max_sample} rows for speed.",
                value=True
            )
        else:
            sample = False

        # 2. Column subset selector for profiling
        st.markdown("Select columns to include in the full profile (optional):")
        selected_columns = st.multiselect(
            "Columns for ydata-profiling (default: all columns)",
            list(pd_df.columns),
            default=list(pd_df.columns)[:min(10, len(pd_df.columns))]
        )

        if st.button("Generate Full Profile Report"):
            prof_df = pd_df
            if sample:
                prof_df = prof_df.sample(n=max_sample, random_state=42)
                st.info(f"Profiling on a random sample of {max_sample} rows for faster results.")
            if selected_columns:
                prof_df = prof_df[selected_columns]
            # Basic ProfileReport call
            profile = ProfileReport(
                prof_df,
                title="Automated Data Profile",
                explorative=True
            )
            st_profile_report(profile)

            
# ---------------- TAB 4:ML ----------------            
with tab_ml:
    # Load the cleaned dataframe fresh from session state if available (passed from cleaning)
    cleaned_df = st.session_state.get('cleaned_df')
    
    if cleaned_df is None:
        st.info("Please clean and prepare your dataset first in the Cleaning tab, then reload this tab.")
        st.stop()
    
    df = cleaned_df  # Use cleaned data only here, no fallback to original
    
    if df.is_empty():
        st.info("Cleaned dataset is empty.")
        st.stop()
    
    # ML Use case selector
    ml_use_case = st.radio(
        "Select a Machine Learning use case:",
        [
            "Credit Default Prediction",
            "Credit Limit Estimation (Regression)"
        ],
        index=0,
        horizontal=True
    )
    
    st.markdown("---")
    st.header("Machine Learning Demo")
    st.subheader(f"Selected Use Case: {ml_use_case}")
    
    if ml_use_case == "Credit Default Prediction":
        required_columns = [
            "TARGET", "CODE_GENDER", "DAYS_BIRTH", "CNT_CHILDREN", "AMT_INCOME_TOTAL",
            "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE", "DAYS_EMPLOYED",
            "FLAG_OWN_CAR", "FLAG_OWN_REALTY", "NAME_HOUSING_TYPE", "ORGANIZATION_TYPE"
        ]
    else:
        required_columns = [
            "AMT_CREDIT", "AMT_INCOME_TOTAL", "DAYS_BIRTH", "CNT_CHILDREN",
            "DAYS_EMPLOYED", "NAME_HOUSING_TYPE", "ORGANIZATION_TYPE"
        ]
    
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        st.error(
            f"Missing required columns: {', '.join(missing_cols)}.\n"
            "Please ensure these columns exist after cleaning."
        )
        st.stop()
    
    pd_df = df.select(required_columns).to_pandas()
    
    # Check missing values
    missing_vals = pd_df.isnull().sum()
    missing_vals = missing_vals[missing_vals > 0]
    if not missing_vals.empty:
        st.warning("The following required features have missing values (fill them in Cleaning tab):")
        st.write(missing_vals)
        st.stop()

    # Numeric columns validations (for “Credit Default Prediction” only)
    if ml_use_case == "Credit Default Prediction":
        numeric_cols = [
            "DAYS_BIRTH", "CNT_CHILDREN", "AMT_INCOME_TOTAL",
            "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE", "DAYS_EMPLOYED"
        ]
        for col in numeric_cols:
            if not pd.api.types.is_numeric_dtype(pd_df[col]):
                st.error(f"Column '{col}' is not numeric. Please fix data types in cleaned data.")
                st.stop()
        if pd_df["TARGET"].isnull().any():
            st.error("Target column `TARGET` has missing values. Fill or remove missing targets.")
            st.stop()

    st.success("Data passes all checks and is ready for ML training.")
    
    st.subheader("Step 2: Train Model")
    
    selected_model = st.selectbox("Select Model:", ["Logistic Regression", "Linear Regression"] if ml_use_case == "Credit Default Prediction" else ["Linear Regression"])
    
    categorical_cols = [
        "CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY", "NAME_HOUSING_TYPE", "ORGANIZATION_TYPE"
    ]
    
    if st.button("Train Model"):
        if ml_use_case == "Credit Default Prediction":
            features = [
                "CODE_GENDER", "DAYS_BIRTH", "CNT_CHILDREN", "AMT_INCOME_TOTAL",
                "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE", "DAYS_EMPLOYED",
                "FLAG_OWN_CAR", "FLAG_OWN_REALTY", "NAME_HOUSING_TYPE", "ORGANIZATION_TYPE"
            ]
            X = pd_df[features].copy()
            y = pd_df["TARGET"].astype(int)
        else:
            features = [
                "CODE_GENDER", "DAYS_BIRTH", "CNT_CHILDREN", "AMT_INCOME_TOTAL",
                "AMT_ANNUITY", "AMT_GOODS_PRICE", "DAYS_EMPLOYED",
                "FLAG_OWN_CAR", "FLAG_OWN_REALTY", "NAME_HOUSING_TYPE", "ORGANIZATION_TYPE"
            ]
            X = pd_df[features].copy()
            y = pd_df["AMT_CREDIT"].astype(float)
        
        # Fill missing values
        for col in X.select_dtypes(include=[np.number]).columns:
            X[col] = X[col].fillna(X[col].median())
        for col in categorical_cols:
            X[col] = X[col].fillna(X[col].mode()[0])
        
        # Encode categoricals
        encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            encoders[col] = le
        
        # Train/Test split
        if ml_use_case == "Credit Default Prediction":
            X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
            model = LogisticRegression(max_iter=500)
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = LinearRegression()

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Save to session for prediction
        st.session_state['model'] = model
        st.session_state['encoders'] = encoders
        st.session_state['X'] = X
        st.session_state['features'] = features

        if ml_use_case == "Credit Default Prediction":
            acc = accuracy_score(y_test, y_pred)
            st.success(f"Model trained! Accuracy on test data: {acc:.3f}")
            cm = confusion_matrix(y_test, y_pred)
            st.write("Confusion Matrix:")
            st.dataframe(pd.DataFrame(cm, columns=['Pred 0', 'Pred 1'], index=['Actual 0', 'Actual 1']))
            report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            report_df = pd.DataFrame(report_dict).transpose().round(2)
            st.markdown("#### Classification Report")
            st.dataframe(report_df, use_container_width=True)
            st.write("Feature Importances (coefficients):")
            st.dataframe(pd.DataFrame({"Feature": features, "Importance": model.coef_[0]}).sort_values(by="Importance", ascending=False))
        else:
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            rmse = mse ** 0.5
            st.success("Model trained successfully!")
            st.write(f"R² score (Goodness of fit): **{r2 * 100:.2f}%**")
            st.write(f"Root Mean Squared Error (RMSE): **₹{rmse:,.0f}** (average prediction error)")

    st.subheader("Step 3: Predict on New Applicant")

    model = st.session_state.get('model')
    encoders = st.session_state.get('encoders')
    X = st.session_state.get('X')
    if model and encoders and X is not None:
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
                if ml_use_case == "Credit Default Prediction":
                    prob = model.predict_proba(input_df)[0][1]
                    pred = model.predict(input_df)
                    st.info(f"Predicted probability of default: {prob:.2%}")
                    st.write(f"Prediction: {'Default Risk' if pred == 1 else 'Low Risk'}")
                else:
                    pred = model.predict(input_df)[0]
                    st.info(f"Predicted Credit Limit for this applicant: **₹{pred:,.0f}**")
            except Exception as e:
                st.error(f"Prediction error: {e}")
    else:
        st.info("Please train the model first.")



# ---------------- TAB 5: Visualizations ----------------
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
