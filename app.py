
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

st.title("Western Wear Analytics Dashboard")

df = pd.read_csv("data.csv")

# Cleaning
df = df.replace("Outlier", np.nan).dropna()

# Encoding
le_dict = {}
for col in df.columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le

# Classification
X = df.drop("Purchase_Intent_30d", axis=1)
y = df["Purchase_Intent_30d"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
clf = DecisionTreeClassifier(max_depth=5)
clf.fit(X_train, y_train)

st.write("Model Accuracy:", clf.score(X_test, y_test))

# Clustering
kmeans = KMeans(n_clusters=4, random_state=42)
df["Cluster"] = kmeans.fit_predict(X)
st.write("Cluster Distribution")
st.write(df["Cluster"].value_counts())

# Prediction
st.subheader("Predict New Customer")

input_data = {}
for col in df.columns:
    if col != "Purchase_Intent_30d":
        input_data[col] = st.selectbox(col, df[col].unique())

if st.button("Predict"):
    input_df = pd.DataFrame([input_data])
    pred = clf.predict(input_df)
    st.success(f"Prediction: {pred[0]}")
