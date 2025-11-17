import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc

from sklearn.cluster import KMeans
from scipy.stats import chi2_contingency, f_oneway

# ======================
# LOAD DATA
# ======================
df = pd.read_csv("Customer-Churn.csv")

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

# Encode churn
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# ======================
# PREPROCESSOR (FIXED)
# ======================
num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
cat_cols = [col for col in df.columns if df[col].dtype == "object"]

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))  # FIXED
])

num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_cols),
    ("cat", cat_pipeline, cat_cols)
])

# ======================
# MODEL TRAINING
# ======================
X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(random_state=42))
])

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

# ROC curve
y_prob = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

# ======================
# STREAMLIT SIDEBAR
# ======================
st.sidebar.title("Navigation")

pages = ["Overview", "EDA", "Clustering", "Models & Feature Importance",
         "Statistical Tests", "Predict"]

choice = st.sidebar.radio("Go to", pages)

# ============================================================
# PAGE 1 — OVERVIEW
# ============================================================
if choice == "Overview":
    st.title("📊 Customer Churn Dashboard")
    st.write("Dataset Preview:")
    st.dataframe(df.head())

    st.write(f"### 🔍 Model Accuracy: **{acc:.2f}**")

    # ROC Curve
    fig = plt.figure(figsize=(6,4))
    plt.plot(fpr, tpr)
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    st.pyplot(fig)

# ============================================================
# PAGE 2 — EDA
# ============================================================
elif choice == "EDA":
    st.title("📊 Exploratory Data Analysis")

    # Churn distribution
    fig = plt.figure(figsize=(5,3))
    sns.countplot(data=df, x="Churn")
    plt.title("Churn Distribution")
    st.pyplot(fig)

    # Correlation heatmap
    numeric_df = df.select_dtypes(include=["int64", "float64"])
    fig = plt.figure(figsize=(8,5))
    sns.heatmap(numeric_df.corr(), cmap="coolwarm")
    plt.title("Correlation Heatmap")
    st.pyplot(fig)

    # Tenure vs churn
    fig = plt.figure(figsize=(5,3))
    sns.boxplot(x="Churn", y="tenure", data=df)
    st.pyplot(fig)

    st.subheader("Missing Values Heatmap")
    fig = plt.figure(figsize=(8,4))
    sns.heatmap(df.isnull(), cbar=False)
    st.pyplot(fig)

# ============================================================
# PAGE 3 — CLUSTERING
# ============================================================
elif choice == "Clustering":
    st.title("🔵 Customer Segmentation")

    segment_data = df[["tenure", "MonthlyCharges", "TotalCharges"]]
    scaled = StandardScaler().fit_transform(segment_data)

    inertia = []
    for k in range(1, 10):
        km = KMeans(n_clusters=k, random_state=42)
        km.fit(scaled)
        inertia.append(km.inertia_)

    fig = plt.figure()
    plt.plot(range(1,10), inertia, "o-")
    plt.title("Elbow Method")
    st.pyplot(fig)

    kmeans = KMeans(n_clusters=3, random_state=42)
    df["Cluster"] = kmeans.fit_predict(scaled)

    st.plotly_chart(px.scatter(
        df, x="MonthlyCharges", y="TotalCharges", color="Cluster",
        title="Customer Clusters"
    ))

# ============================================================
# PAGE 4 — MODEL & FEATURE IMPORTANCE
# ============================================================
elif choice == "Models & Feature Importance":
    st.title("📌 Model Performance & Feature Importance")

    st.write("### Classification Report")
    st.text(classification_report(y_test, y_pred))

    # Feature Importance
    rf_model = model.named_steps["model"]
    importances = rf_model.feature_importances_

    fig = plt.figure(figsize=(6,4))
    plt.bar(range(len(importances)), importances)
    plt.title("Random Forest Feature Importance")
    st.pyplot(fig)

# ============================================================
# PAGE 5 — STATISTICAL TESTS
# ============================================================
elif choice == "Statistical Tests":
    st.title("📑 Statistical Tests")

    st.subheader("Chi-Square Tests (Categorical)")
    for col in cat_cols:
        contingency = pd.crosstab(df[col], df["Churn"])
        chi2, p, _, _ = chi2_contingency(contingency)
        st.write(f"{col}: p = {p:.4f}")

    st.subheader("ANOVA (Numerical)")
    for col in num_cols:
        groups = [group[col] for _, group in df.groupby("Churn")]
        stat, p = f_oneway(*groups)
        st.write(f"{col}: p = {p:.4f}")

# ============================================================
# PAGE 6 — PREDICT
# ============================================================
elif choice == "Predict":
    st.title("🎯 Predict Customer Churn")

    user_input = {}
    for col in X.columns:
        if col in num_cols:
            user_input[col] = st.number_input(col, value=float(df[col].median()))
        else:
            user_input[col] = st.selectbox(col, df[col].unique())

    if st.button("Predict"):
        inp = pd.DataFrame([user_input])
        pred = model.predict(inp)[0]
        st.subheader("Prediction Result:")
        st.write("🔴 Will Churn" if pred == 1 else "🟢 Will Not Churn")
