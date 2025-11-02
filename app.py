import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px
import matplotlib.pyplot as plt

# Streamlit Page Config
st.set_page_config(page_title="Soil Analysis System", layout="wide", page_icon="🌱")

# Custom CSS for Styling
st.markdown("""
    <style>
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
        font-weight: bold;
        border-radius: 12px;
        padding: 10px 20px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stRadio>label, .stSelectbox>label {
        font-size: 18px;
        color: #4CAF50;
        font-weight: bold;
    }
    .stTextInput>label {
        font-size: 18px;
        color: #4CAF50;
        font-weight: bold;
    }
    .stFileUploader {
        background-color: #f0f0f0;
        border-radius: 10px;
        padding: 20px;
    }
    .stMetric>div {
        background-color: #e0f7fa;
        border-radius: 12px;
        padding: 10px;
    }
    .header {
        color: #4CAF50;
        text-align: center;
    }
    .good { color: green; font-weight: bold; }
    .moderate { color: orange; font-weight: bold; }
    .bad { color: red; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("Soil Health Analysis System")
    mode = st.radio("Select Task Mode:", ["Classification", "Regression"], index=0)
    uploaded_file = st.file_uploader("Upload Soil Data (CSV)", type=["csv"])

# Main Section
if uploaded_file:
    # Load and display the dataset
    data = pd.read_csv(uploaded_file)
    st.write("### Data Preview", data.head())

    # Preprocess Data (Scale and clean)
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    scaler = MinMaxScaler()
    data[numeric_cols] = scaler.fit_transform(data[numeric_cols])
    st.write("### Scaled Data", data.head())

    # Feature and target selection
    target = st.selectbox("Select the target variable", options=data.columns)

    # Train-test split
    X = data.drop(columns=[target])
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if mode == "Classification":
        model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    else:
        model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)

    # Train Model
    if st.button("Train Model"):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Display Model Results
        if mode == "Classification":
            acc = accuracy_score(y_test, y_pred)
            st.metric("Accuracy", f"{acc*100:.2f}%")
            # Color-coded output based on accuracy
            if acc > 0.8:
                st.markdown('<h3 class="good">Soil health is Excellent. Suitable for various crops!</h3>', unsafe_allow_html=True)
            elif acc > 0.5:
                st.markdown('<h3 class="moderate">Soil health is Moderate. Further analysis needed.</h3>', unsafe_allow_html=True)
            else:
                st.markdown('<h3 class="bad">Soil health is Poor. Improvement is needed.</h3>', unsafe_allow_html=True)

            # Feature Importance Visualization
            fi = pd.DataFrame({
                'Feature': X.columns,
                'Importance': model.feature_importances_
            }).sort_values(by='Importance', ascending=False)
            st.bar_chart(fi.set_index('Feature'))
        else:
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            st.metric("RMSE", f"{rmse:.2f}")
            r2 = r2_score(y_test, y_pred)
            st.metric("R²", f"{r2:.2f}")
            # Crop suggestion based on regression model
            if r2 > 0.8:
                st.markdown('<h3 class="good">Soil is optimal for crop growth.</h3>', unsafe_allow_html=True)
            elif r2 > 0.5:
                st.markdown('<h3 class="moderate">Soil is moderate, improve fertility for better yield.</h3>', unsafe_allow_html=True)
            else:
                st.markdown('<h3 class="bad">Soil requires significant improvement.</h3>', unsafe_allow_html=True)

            # Residual Plot
            residuals = y_test - y_pred
            fig, ax = plt.subplots()
            ax.hist(residuals, bins=50)
            ax.set_title("Residual Distribution")
            ax.set_xlabel("Residuals")
            ax.set_ylabel("Frequency")
            st.pyplot(fig)

    # Crop Suggestion Logic based on Soil Health (using random ranges for demonstration)
    st.write("### Crop Suggestions")
    if mode == "Classification":
        if acc > 0.8:
            st.write("✅ **Ideal crops**: Wheat, Rice, Corn")
        elif acc > 0.5:
            st.write("⚠️ **Suitable crops**: Beans, Peas, Potatoes")
        else:
            st.write("🚫 **Poor crops for this soil**: Avoid high-yield crops")
    else:
        if r2 > 0.8:
            st.write("✅ **Optimal crops**: Wheat, Sunflower, Corn")
        elif r2 > 0.5:
            st.write("⚠️ **Considerable crops**: Barley, Oats")
        else:
            st.write("🚫 **Inappropriate for most crops**")

    # Display Important Features
    feature_importances = model.feature_importances_
    st.write("### Most Important Soil Features")
    important_features = pd.DataFrame({
        'Feature': X.columns,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)
    st.dataframe(important_features)

# Footer
st.markdown("### Developed for Sustainable Agriculture Solutions")
