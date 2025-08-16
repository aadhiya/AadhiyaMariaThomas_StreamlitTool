## Streamlit Data Tool:  For Cleansing, Profiling & ML train and predict by Aadhiya Thomas

This Streamlit-based app provides a unified interface for uploading, cleaning, profiling, analyzing, and modeling tabular data from S3 using Streamlit.

It supports fast, interactive data science workflows with advanced handling for both small and large datasets.
________________________________________

## Stack Choices: Polars vs Pandas

Polars and Pandas are both supported for tabular data processing:

 **Polars:** This is used as the default engine for data manipulation, preview, and cleaning. It offers significantly faster performance on large datasets, lower memory usage, and powerful lazy evaluation.
 
•	Most internal operations (reading CSV/Excel, cleaning, etc.) route through Polars whenever possible.

•	Automatic fallback to Pandas for unsupported formats or operations.

**Pandas:** This is used for interoperation with ML libraries (scikit-learn), visualization (matplotlib, seaborn), and profiling.

•	Conversion between Polars and Pandas is seamless.

In this app for best performance, I have kept working within Polars until ML/training/modeling steps.

________________________________________

## Data Profiling Integration

Automated Data Profiling is provided via these integrations:

•	Custom Numeric/Categorical Summaries: With Fast, in-app previews using Polars for basic statistics (mean, median, std, value counts).

•	Streamlit-Pandas-Profiling: Which provides Embedded profiling reports which are displayed directly in Streamlit, allowing interactive exploration without leaving the app.

•	ydata-profiling (formerly pandas-profiling):
•	Full exploratory profile reports for your dataset, produced instantly.
•	Supports interactive sampling for large datasets (with toggles in the UI to profile up to 1,000 random rows for rapid diagnostics).
•	You can select which columns are included in profiles for targeted analysis.





## Libraries used in this project  and their purpose
**Streamlit**

It Lets you build interactive web apps easily with Python, especially for data-focused apps.
I have used it here to create the graphical interface where users upload data, see results, and interact with the tool.

Reference: https://streamlit.io/

**Polars**

It is a very fast tool to work with large tables of data, alternative to pandas, especially useful when data is big.
Here i have used it to handle large datasets quickly and efficiently in the app.

Reference: https://pola.rs/

**pandas**

It is a popular data analysis tool in Python for handling and analyzing tables of data.
I am using it here as a fallback and for compatibility with tools that require pandas data formats.

Reference: https://pandas.pydata.org/

**scikit-learn**

It provides many machine learning algorithms to build models that can predict or classify data.
I have used it here to train models and make predictions from the data within the app.

Reference: https://scikit-learn.org/

**matplotlib**

It is a library to create charts and graphs from data.
I have used this to visualize data trends and model results.

Reference: https://matplotlib.org/

**seaborn**

It builds on matplotlib to create attractive and easy-to-understand statistical graphics.
I used this to create more visually appealing and informative data visualizations.

Reference: https://seaborn.pydata.org/

**ydata-profiling**

It automatically analyzes and summarizes data to give very detailed reports about the dataset (like how many missing values, distributions, and correlations).
I have used it to generate comprehensive data profiling reports inside the app quickly.

Reference: https://ydata-profiling.ydata.ai/

**pyarrow**

This provides fast data interchange and storage tools, useful for handling large datasets efficiently.
Used this to support faster data processing and integration with polars or other backends.

Reference: https://arrow.apache.org/docs/python/

## Data Profiling & Exploration Module 

This module provides an interactive and comprehensive overview of any uploaded CSV dataset, including:

**Descriptive statistics for numeric columns:**

Calculates and displays mean, median, standard deviation, minimum, maximum, and missing value counts for all numeric features.

**Distributions and counts for categorical data:**

Provides value counts for each categorical column, allowing easy inspection of category frequencies with expandable views.

**Missing values detection and summary:**

Shows total missing/null counts per column in an easy-to-read table format for quick data quality assessment.

**Interactive column-level exploration:**

Enables users to select any column and explore its distribution visually – histograms for numeric columns and bar charts for categorical columns.

All profiling features are toggled on-demand via the sidebar to keep the app interface clean and user-friendly.
