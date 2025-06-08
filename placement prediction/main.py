import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from itertools import combinations

st.title("🎓 Placement Prediction and Analysis Web App")

# Load dataset
@st.cache_data
def load_data():
    data = pd.read_csv('placement_data.csv')
    label_encoder = LabelEncoder()
    for col in ['Internship', 'ExtraCurricular', 'Placement', 'Branch', 'Gender', 'Certifications']:
        data[col] = label_encoder.fit_transform(data[col])
    return data

data = load_data()
st.subheader("📊 Dataset Preview")
st.dataframe(data.head())

# Feature selection
feature_cols = ['CGPA', 'Internship', 'ExtraCurricular', 'Branch', 'Gender',
                'Backlogs', 'Certifications', 'CommunicationSkills', 'ProjectCount']
X = data[feature_cols]
y = data['Placement']

# Scale and split
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Train models
models = {
    'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Support Vector Machine': SVC(kernel='rbf', random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, random_state=42),
    'K Nearest Neighbors': KNeighborsClassifier(n_neighbors=5)
}

results = {}
st.subheader("🔍 Model Evaluation")
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy
    st.write(f"### {name}")
    st.write("Accuracy:", accuracy)
    st.text("Confusion Matrix:")
    st.write(confusion_matrix(y_test, y_pred))
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))

# Accuracy comparison plot
st.subheader("📈 Model Accuracy Comparison")
plt.figure(figsize=(8, 4))
plt.bar(results.keys(), results.values(), color='skyblue')
plt.ylabel('Accuracy')
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot()

# Correlation heatmap
st.subheader("🧠 Feature Correlation Heatmap")
plt.figure(figsize=(10, 6))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
st.pyplot()

# CGPA distribution
st.subheader("📚 CGPA Distribution")
sns.histplot(data['CGPA'], bins=20, kde=True, color='purple')
st.pyplot()

# Placement count
st.subheader("✅ Placement Count")
sns.countplot(x='Placement', data=data, palette='Set2')
st.pyplot()

# Placed students
st.subheader("🎉 Placed Students")
placed_students = data[data['Placement'] == 1]
st.write(f"Total placed students: {len(placed_students)}")
st.dataframe(placed_students.head(10))
placed_students.to_csv("placed_students_list.csv", index=False)
st.success("Saved full list to 'placed_students_list.csv'")

# Predict placement for new student using toggles
st.subheader("🤖 Predict Placement for a New Student")

with st.form("placement_form"):
    cgpa = st.slider("📊 CGPA", 0.0, 10.0, 7.0)
    internship = st.checkbox("📄 Completed Internship?")
    extra = st.checkbox("🎨 Participated in Extra Curricular Activities?")

    branch_option = st.selectbox("💻 Branch", options=["CSE", "IT", "ECE"])
    branch = {"CSE": 0, "IT": 1, "ECE": 2}[branch_option]

    gender_option = st.radio("🧑‍🤝‍🧑 Gender", options=["Male", "Female"])
    gender = {"Male": 0, "Female": 1}[gender_option]

    backlogs = st.slider("📚 Number of Backlogs", 0, 10, 0)
    cert = st.checkbox("🏅 Holds Certifications?")
    comm = st.slider("🗣️ Communication Skills (1-10)", 1, 10, 7)
    proj = st.slider("🧪 Project Count", 0, 10, 2)

    submitted = st.form_submit_button("🎯 Predict Placement")

if submitted:
    input_data = np.array([[
        cgpa,
        int(internship),
        int(extra),
        branch,
        gender,
        backlogs,
        int(cert),
        comm,
        proj
    ]], dtype=np.float64)

    scaled_input = scaler.transform(input_data)
    prediction = models['Random Forest'].predict(scaled_input)

    st.markdown("---")
    if prediction[0] == 1:
        st.success("✅ Based on your profile, you are likely to be **PLACED**!")
    else:
        st.error("❌ Based on your profile, you are **not likely to be placed**. Consider improving your skills or profile.")

# Backtracking feature selection
st.subheader("🔬 Best Feature Combination (Backtracking)")
def backtrack_feature_selection(features, target, max_features):
    best_score = 0
    best_combo = None
    for r in range(1, max_features + 1):
        for combo in combinations(features.columns, r):
            X_temp = features[list(combo)]
            X_train_temp, X_test_temp, y_train_temp, y_test_temp = train_test_split(X_temp, target, test_size=0.3, random_state=42)
            model = RandomForestClassifier(n_estimators=200, random_state=42)
            model.fit(X_train_temp, y_train_temp)
            score = model.score(X_test_temp, y_test_temp)
            if score > best_score:
                best_score = score
                best_combo = combo
    return best_combo, best_score

best_feat, best_acc = backtrack_feature_selection(data[feature_cols], y, max_features=3)
st.write(f"Best feature combination: {best_feat}")
st.write(f"Accuracy with best combination: {best_acc}")
