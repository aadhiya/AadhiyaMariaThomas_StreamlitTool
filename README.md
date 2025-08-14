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

