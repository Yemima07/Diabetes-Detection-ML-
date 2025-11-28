import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, make_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB

# Streamlit page setup
st.set_page_config(page_title="Diabetes Detection ML App", layout="wide")
st.title("🩺 Diabetes Detection Machine Learning Dashboard")

# Load dataset
df = pd.read_csv("diabetes.csv")
st.subheader("Dataset Preview")
st.write(df.head())

# Dataset info
st.subheader("Dataset Summary")
st.write(df.describe())
st.write("Null values present:", df.isnull().values.any())

# Histogram
st.subheader("Feature Distributions")
fig, ax = plt.subplots(figsize=(10, 10))
df.hist(bins=10, ax=ax)
st.pyplot(fig)

# Correlation heatmap
st.subheader("Correlation Heatmap")
fig, ax = plt.subplots()
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)

# Outcome count
st.subheader("Outcome Count (0 = No Diabetes, 1 = Diabetes)")
fig, ax = plt.subplots()
sns.countplot(y=df['Outcome'], palette="Set1", ax=ax)
st.pyplot(fig)

# Outlier removal
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
df_out = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]
st.write("Original shape:", df.shape, "After outlier removal:", df_out.shape)

# Pairplot after outlier removal
st.subheader("Pairplot After Outlier Removal")
st.pyplot(sns.pairplot(df_out, hue="Outcome"))

# Train-test split
X = df_out.drop(columns=["Outcome"])
y = df_out["Outcome"]
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)

# Custom scoring
def tn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 0]
def fp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 1]
def fn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 0]
def tp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 1]

scoring = {'tp': make_scorer(tp), 'tn': make_scorer(tn),
           'fp': make_scorer(fp), 'fn': make_scorer(fn)}

# Models
models = {
    "Logistic Regression": LogisticRegression(),
    "SVM (Linear Kernel)": SVC(kernel="linear", probability=True),
    "KNN (k=3)": KNeighborsClassifier(n_neighbors=3),
    "Random Forest": RandomForestClassifier(),
    "Naive Bayes": GaussianNB(),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=50, learning_rate=0.2)
}

acc = []
roc = []

st.subheader("Model Performance")

for name, clf in models.items():
    clf.fit(train_X, train_y)
    y_pred = clf.predict(test_X)
    ac = accuracy_score(test_y, y_pred)
    rc = roc_auc_score(test_y, y_pred)
    acc.append(ac)
    roc.append(rc)
    st.write(f"**{name}** → Accuracy: {ac:.3f}, ROC AUC: {rc:.3f}")

# Accuracy bar plot
st.subheader("Accuracy Comparison")
fig, ax = plt.subplots(figsize=(9, 4))
ax.bar(models.keys(), acc, color="skyblue")
ax.set_ylabel("Accuracy Score")
ax.set_xticklabels(models.keys(), rotation=45, ha="right")
st.pyplot(fig)

# ROC AUC bar plot
st.subheader("ROC AUC Comparison")
fig, ax = plt.subplots(figsize=(9, 4))
ax.bar(models.keys(), roc, color="salmon")
ax.set_ylabel("ROC AUC")
ax.set_xticklabels(models.keys(), rotation=45, ha="right")
st.pyplot(fig)