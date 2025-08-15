import streamlit as st
import polars as pl
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import io
import locale
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_squared_error, r2_score

# ============================= PAGE CONFIG =============================
st.set_page_config(
    page_title="Aadhiya Maria Thomas - Streamlit Data Tool",
    layout="wide"
)
st.title("📊 Streamlit Data Tool: Cleansing, Profiling & ML")

numeric_polars_types = [
    pl.Int8, pl.Int16, pl.Int32, pl.Int64,
    pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
    pl.Float32, pl.Float64
]

# ============================= SIDEBAR =============================
st.sidebar.header("📂 File Upload")
uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])
df = None

if uploaded_file:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pl.read_csv(uploaded_file, infer_schema_length=10000, ignore_errors=True)
        else:
            df = pl.from_pandas(pd.read_excel(uploaded_file))
        st.sidebar.success(f"{df.height} rows × {df.width} columns loaded.")
    except Exception as e:
        st.sidebar.error(f"Error loading file: {e}")

# Global Sidebar Toggles
st.sidebar.markdown("---")
show_profiling = st.sidebar.checkbox("📈 Show Data Profiling")
#show_ml_demo = st.sidebar.checkbox("🤖 Show ML Use Case Demo")

# Export Cleaned CSV
if df is not None:
    buf = io.BytesIO()
    df.write_csv(buf)
    buf.seek(0)
    st.sidebar.download_button(
        "💾 Download Cleaned CSV",
        buf,
        file_name="cleaned_data.csv",
        mime="text/csv"
    )

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
    st.subheader("📥 Dataset Preview")
    if df is not None:
        st.write(f"**Shape:** {df.height} rows × {df.width} columns")
        st.dataframe(df.head(10), use_container_width=True)
    else:
        st.info("Please upload a dataset from the sidebar to begin.")

# ---------------- TAB 2: Cleaning ----------------
with tab_clean:
    if df is None:
        st.info("Upload a dataset to start cleaning.")
    else:
        st.subheader("🧹 Data Cleaning")
        
        # Missing Value Summary
        st.markdown("### Missing Data Summary")
        missing_counts = {col: int(df[col].null_count()) for col in df.columns}
        st.table(missing_counts)

        # Missing Value Handling
        numeric_cols = [c for c, t in zip(df.columns, df.dtypes) if t in numeric_polars_types]
        selected_cols = st.multiselect("Select column(s) to clean:", df.columns)
        apply_all = st.checkbox("Apply to ALL numeric columns")

        if apply_all:
            selected_cols = numeric_cols

        method = st.radio("Choose method:", ["Fill with Mean", "Fill with Median", "Fill with Mode", "Drop Rows", "Drop Columns"])

        if st.button("Apply Missing Value Handling"):
            if not selected_cols:
                st.warning("Select columns first.")
            else:
                for c in selected_cols:
                    if method == "Fill with Mean":
                        df = df.with_columns(pl.col(c).fill_null(df[c].mean()))
                    elif method == "Fill with Median":
                        df = df.with_columns(pl.col(c).fill_null(df[c].median()))
                    elif method == "Fill with Mode":
                        mode_val = df[c].drop_nulls().mode()[0]
                        df = df.with_columns(pl.col(c).fill_null(mode_val))
                    elif method == "Drop Rows":
                        df = df.drop_nulls(subset=selected_cols)
                    elif method == "Drop Columns":
                        df = df.drop(selected_cols)
                st.success("Missing values handled successfully.")

        # Duplicate Removal
        duplicates = df.to_pandas().duplicated().sum()
        st.write(f"Found **{duplicates}** duplicate rows.")
        if duplicates > 0 and st.button("Remove Duplicates"):
            df = df.unique()
            st.success("Duplicates removed.")

# ---------------- TAB 3: Profiling ----------------
with tab_profile:
    if df is None:
        st.info("Upload to view profiling.")
    elif not show_profiling:
        st.info("Enable profiling from sidebar.")
    else:
        st.subheader("📈 Data Profiling")
        
        # Numeric Columns
        num_cols = [c for c, t in zip(df.columns, df.dtypes) if t in numeric_polars_types]
        if num_cols:
            stats_df = df.select(
                [pl.col(c).mean().alias(f"{c}_mean") for c in num_cols] +
                [pl.col(c).median().alias(f"{c}_median") for c in num_cols] +
                [pl.col(c).std().alias(f"{c}_std") for c in num_cols]
            ).to_pandas()
            st.dataframe(stats_df)
        else:
            st.warning("No numeric columns found.")

        # Categorical
        cat_cols = [c for c, t in zip(df.columns, df.dtypes) if t == pl.Utf8]
        for col in cat_cols:
            counts = df.select(pl.col(col).value_counts().sort(descending=True)).to_pandas()
            with st.expander(f"Value Counts: {col}"):
                st.dataframe(counts)

with tab_ml:
    if df is None:
        st.info("Upload a dataset first.")
    else:
        # ML Use Case selector directly in main tab
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
            st.markdown("**Goal:** Predict the risk that an applicant will miss scheduled payments on their loan or credit obligation based on their demographic features (like gender, age, number of children), financial features (income, credit amount, annuity, goods price), employment duration, ownership flags, and housing and organization types.")
            st.subheader("Step 1: Prepare Data Quality Check")
            required_columns = [
                "TARGET", "CODE_GENDER", "DAYS_BIRTH", "CNT_CHILDREN", "AMT_INCOME_TOTAL",
                "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE", "DAYS_EMPLOYED",
                "FLAG_OWN_CAR", "FLAG_OWN_REALTY", "NAME_HOUSING_TYPE", "ORGANIZATION_TYPE"
            ]

            missing_cols = [col for col in required_columns if col not in df.columns]
            err_msgs = []

            if missing_cols:
                err_msgs.append(
                    f"Missing required columns: {', '.join(missing_cols)}.\n"
                    "Use 'Interactive Column Explorer' to check columns and 'Column Explanation' for details."
                )
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
                        err_msgs.append(
                            f"Column '{col}' is not numeric. Use 'Interactive Column Explorer' and "
                            "'Encode Categorical Feature(s)' or fix data type in sidebar."
                        )

                if pd_df["TARGET"].isnull().any():
                    err_msgs.append("Target column `TARGET` has missing values. Use sidebar cleaning tools to drop or fill.")

            if len(err_msgs) > 0 or (not missing_cols and not num_missing.empty):
                st.error("Data is NOT ready for ML training.")
                for msg in err_msgs:
                    st.markdown(msg)
                st.info("Please use the data cleaning options in the sidebar, then export and reload the cleaned data.")
                st.stop()
            else:
                st.success("Data passes all checks and is ready for ML training.")

                st.subheader("Step 2: Train Model")
                selected_model = st.selectbox("Select Model:", ["Logistic Regression"])

                categorical_cols = [
                    "CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY",
                    "NAME_HOUSING_TYPE", "ORGANIZATION_TYPE"
                ]

                if st.button("Train Model"):
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

                    encoders = {}
                    for col in categorical_cols:
                        le = LabelEncoder()
                        X[col] = le.fit_transform(X[col].astype(str))
                        encoders[col] = le

                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, stratify=y, test_size=0.2, random_state=42
                    )

                    model = LogisticRegression(max_iter=500)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    acc = accuracy_score(y_test, y_pred)

                    st.session_state['model'] = model
                    st.session_state['encoders'] = encoders
                    st.session_state['X'] = X

                    st.success(f"Model trained! Accuracy on test data: {acc:.3f}")

                    cm = confusion_matrix(y_test, y_pred)
                    st.write("Confusion Matrix:")
                    st.dataframe(pd.DataFrame(cm, columns=['Pred 0', 'Pred 1'], index=['Actual 0', 'Actual 1']))

                    report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
                    report_df = pd.DataFrame(report_dict).transpose()
                    report_df = report_df.round(2)
                    st.markdown("#### Classification Report")
                    st.dataframe(report_df, use_container_width=True)

                    st.write("Feature Importances (coefficients):")
                    st.dataframe(
                        pd.DataFrame({"Feature": features, "Importance": model.coef_[0]})
                        .sort_values(by="Importance", ascending=False)
                    )

                st.subheader("Step 3: Predict Default Risk on New Applicant")

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
                            prob = model.predict_proba(input_df)[0][1]
                            pred = model.predict(input_df)
                            st.info(f"Predicted probability of default: {prob:.2%}")
                            st.write(f"Prediction: {'Default Risk' if pred == 1 else 'Low Risk'}")
                        except Exception as e:
                            st.error(f"Prediction error: {e}")
                else:
                    st.info("Please train the model first.")

        elif ml_use_case == "Credit Limit Estimation (Regression)":
            st.markdown("**Goal:** Predict the expected credit limit for an applicant using key profile features such as income, age, number of children, employment duration, housing status, and organization type")
            st.subheader("Step 1: Prepare Data Quality Check")
            required_columns = [
                "AMT_CREDIT", "AMT_INCOME_TOTAL", "DAYS_BIRTH", "CNT_CHILDREN",
                "DAYS_EMPLOYED", "NAME_HOUSING_TYPE", "ORGANIZATION_TYPE"
            ]

            missing_cols = [col for col in required_columns if col not in df.columns]
            err_msgs = []

            if missing_cols:
                err_msgs.append(
                    f"Missing required columns: {', '.join(missing_cols)}.\n"
                    "Use 'Interactive Column Explorer' to check columns and 'Column Explanation' for details."
                )
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
                        err_msgs.append(
                            f"Column '{col}' is not numeric. Use 'Interactive Column Explorer' and "
                            "'Encode Categorical Feature(s)' or fix data type in sidebar."
                        )

            if len(err_msgs) > 0 or (not missing_cols and not num_missing.empty):
                st.error("Data is NOT ready for ML training.")
                for msg in err_msgs:
                    st.markdown(msg)
                st.info("Please use the data cleaning options in the sidebar, then export and reload the cleaned data.")
                st.stop()
            else:
                st.success("Data passes all checks and is ready for ML training.")

                st.subheader("Step 2: Train Model")
                selected_model = st.selectbox("Select Model:", ["Linear Regression"])

                categorical_cols = [
                    "CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY",
                    "NAME_HOUSING_TYPE", "ORGANIZATION_TYPE"
                ]

                if st.button("Train Model"):
                    features = [
                        "CODE_GENDER", "DAYS_BIRTH", "CNT_CHILDREN", "AMT_INCOME_TOTAL",
                        "AMT_ANNUITY", "AMT_GOODS_PRICE", "DAYS_EMPLOYED",
                        "FLAG_OWN_CAR", "FLAG_OWN_REALTY", "NAME_HOUSING_TYPE", "ORGANIZATION_TYPE"
                    ]

                    X = pd_df[features].copy()
                    y = pd_df["AMT_CREDIT"].astype(float)

                    for col in X.select_dtypes(include=[np.number]).columns:
                        X[col] = X[col].fillna(X[col].median())
                    for col in X.select_dtypes(include='object').columns:
                        X[col] = X[col].fillna(X[col].mode()[0])

                    encoders = {}
                    for col in categorical_cols:
                        le = LabelEncoder()
                        X[col] = le.fit_transform(X[col].astype(str))
                        encoders[col] = le

                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42
                    )
                    model = LinearRegression()
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    mse = mean_squared_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    rmse = mse ** 0.5
                    # Save to session state for later prediction
                    st.session_state['model'] = model
                    st.session_state['encoders'] = encoders
                    st.session_state['X'] = X
                    st.session_state['features'] = features
                    st.success("Model trained successfully!")
                    st.write(f"R² score (Goodness of fit): **{r2 * 100:.2f}%**")
                    st.write(f"Root Mean Squared Error (RMSE): **₹{rmse:,.0f}** (average prediction error)")

                st.subheader("Step 3: Predict Credit Limit for New Applicant")

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
                            pred = model.predict(input_df)[0]

                            predicted_limit = pred
                            st.info(f"Predicted Credit Limit for this applicant: **₹{predicted_limit:,.0f}**")
                        except Exception as e:
                            st.error(f"Prediction error: {e}")

                else:
                    st.info("Please train the model first.")


# ---------------- TAB 5: Visualizations ----------------
with tab_viz:
    if df is None:
        st.info("Upload data to plot.")
    else:
        st.subheader("Visualizations")
        viz_type = st.radio("Choose Visualization:", ["Histogram", "Bar Chart", "Correlation Heatmap"], horizontal=True)

        # ----------- Histogram (Single & Advanced) -----------
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
