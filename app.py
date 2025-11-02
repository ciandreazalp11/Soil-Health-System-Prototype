import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score, confusion_matrix, mean_absolute_error
from io import BytesIO
import joblib
import time
import base64
from PIL import Image
import io as sysio
import os

# ----------------- THEMES -----------------
theme_classification = {
    "background_main": "linear-gradient(120deg, #0f2c2c 0%, #1a4141 40%, #0e2a2a 100%)",
    "sidebar_bg": "rgba(15, 30, 30, 0.95)",
    "primary_color": "#81c784",
    "secondary_color": "#a5d6a7",
    "button_gradient": "linear-gradient(90deg, #66bb6a, #4caf50)",
    "button_text": "#0c1d1d",
    "header_glow_color_1": "#81c784",
    "header_glow_color_2": "#4caf50",
    "menu_icon_color": "#81c784",
    "nav_link_color": "#e0ffe0",
    "nav_link_selected_bg": "#4caf50",
    "info_bg": "#214242",
    "info_border": "#4caf50",
    "success_bg": "#2e5c2e",
    "success_border": "#81c784",
    "warning_bg": "#5c502e",
    "warning_border": "#dcd380",
    "error_bg": "#5c2e2e",
    "error_border": "#ef9a9a",
    "text_color": "#e0ffe0",
    "title_color": "#a5d6a7",
}

theme_sakura = {
    "background_main": "linear-gradient(120deg, #2b062b 0%, #3b0a3b 50%, #501347 100%)",
    "sidebar_bg": "linear-gradient(180deg, rgba(30,8,30,0.95), rgba(45,10,45,0.95))",
    "primary_color": "#ff8aa2",
    "secondary_color": "#ffc1d3",
    "button_gradient": "linear-gradient(90deg, #ff8aa2, #ff3b70)",
    "button_text": "#1f0f16",
    "header_glow_color_1": "#ff93b0",
    "header_glow_color_2": "#ff3b70",
    "menu_icon_color": "#ff93b0",
    "nav_link_color": "#ffd6e0",
    "nav_link_selected_bg": "#ff3b70",
    "info_bg": "#40132a",
    "info_border": "#ff93b0",
    "success_bg": "#3a1b2a",
    "success_border": "#ff93b0",
    "warning_bg": "#3b2530",
    "warning_border": "#ffb3b3",
    "error_bg": "#3a1a22",
    "error_border": "#ff9aa3",
    "text_color": "#ffeef8",
    "title_color": "#ffd6e0",
}

# ----------------- Initialize Theme -----------------
if "current_theme" not in st.session_state:
    # Set default theme if not present
    st.session_state["current_theme"] = theme_classification  # Or your preferred default theme

# Ensure the default values are set for other session state variables
if "df" not in st.session_state:
    st.session_state["df"] = None
if "results" not in st.session_state:
    st.session_state["results"] = None
if "model" not in st.session_state:
    st.session_state["model"] = None
if "scaler" not in st.session_state:
    st.session_state["scaler"] = None
if "task_mode" not in st.session_state:
    st.session_state["task_mode"] = "Classification"  # default task mode
if "trained_on_features" not in st.session_state:
    st.session_state["trained_on_features"] = None

# ----------------- page config -----------------
st.set_page_config(
    page_title="Machine Learning-Driven Soil Analysis for Sustainable Agriculture System",
    layout="wide",
    page_icon="🌿"
)

# ----------------- SIDEBAR (modeling first) -----------------
with st.sidebar:
    st.markdown(
        f"""
        <div class="sidebar-header">
          <h2 class="sidebar-title">🌱 Soil Health System</h2>
          <div class="sidebar-sub">ML-Driven Soil Analysis</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.write("---")
    selected = option_menu(
        None,
        ["🏠 Home", "🤖 Modeling", "📊 Visualization", "📈 Results", "🌿 Insights", "👤 About"],
        icons=["house", "robot", "bar-chart", "graph-up", "lightbulb", "person-circle"],
        menu_icon="list",
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
            "icon": {"color": st.session_state["current_theme"]["menu_icon_color"], "font-size": "18px"},
            "nav-link": {"font-size": "15px", "text-align": "left", "margin": "0px"},
            "nav-link-selected": {"background-color": st.session_state["current_theme"]["nav_link_selected_bg"]},
        }
    )
    st.write("---")
    st.markdown(f"<div style='font-size:12px;color:{st.session_state['current_theme']['text_color']};opacity:0.85'>Developed for sustainable agriculture</div>", unsafe_allow_html=True)

# Model toggle for classification vs regression
def switch_model_mode():
    st.markdown("#### Model Mode")
    chk = st.checkbox("Switch to Regression mode", value=(st.session_state["task_mode"] == "Regression"))
    if chk:
        st.session_state["task_mode"] = "Regression"
        st.session_state["current_theme"] = theme_sakura
    else:
        st.session_state["task_mode"] = "Classification"
        st.session_state["current_theme"] = theme_classification
    apply_theme(st.session_state["current_theme"])

# Visualization section (new model output suggestions)
def show_crop_suggestions(prediction):
    if prediction == "Rice":
        st.markdown('<div style="background-color: #4CAF50; color: white; padding: 10px;">Sustainable</div>', unsafe_allow_html=True)
        st.write("**Rice** is a great choice for humid, loamy soils.")
    elif prediction == "Wheat":
        st.markdown('<div style="background-color: orange; color: white; padding: 10px;">Moderate</div>', unsafe_allow_html=True)
        st.write("**Wheat** grows well in moderate conditions.")
    else:
        st.markdown('<div style="background-color: red; color: white; padding: 10px;">Unsustainable</div>', unsafe_allow_html=True)
        st.write("**This crop may not be suitable for the current conditions.")

# ----------------- HOME -----------------
if page == "🏠 Home":
    st.title("Machine Learning-Driven Soil Analysis for Sustainable Agriculture System")
    st.write("---")
    st.markdown("### 📂 Upload Soil Data")
    upload_and_preprocess_widget()

# ----------------- MODELING -----------------
elif page == "🤖 Modeling":
    st.title("🤖 Modeling — Random Forest")
    if st.session_state["df"] is None:
        st.info("Please upload a dataset first in 'Home'.")
    else:
        df = st.session_state["df"].copy()

        switch_model_mode()  # Adds the toggle to switch between classification and regression

        if st.session_state["task_mode"] == "Classification":
            if 'Nitrogen' in df.columns:
                df['Fertility_Level'] = create_fertility_label(df, col='Nitrogen', q=3)
            y = df['Fertility_Level'] if 'Fertility_Level' in df.columns else None
        else:
            y = df['Nitrogen'] if 'Nitrogen' in df.columns else None

        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'Nitrogen' in numeric_features:
            numeric_features.remove('Nitrogen')

        st.subheader("Feature Selection")
        selected_features = st.multiselect("Features", options=numeric_features, default=numeric_features)

        if not selected_features:
            st.warning("Select at least one feature.")
        else:
            X = df[selected_features]

            # Hyperparameters (for Random Forest)
            st.subheader("Hyperparameters")
            col1, col2 = st.columns(2)
            with col1:
                n_estimators = st.slider("n_estimators", 50, 500, 150, step=50)
            with col2:
                max_depth = st.slider("max_depth", 2, 50, 12)

            # Data Preprocessing & Model Training
            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(X)
            X_scaled_df = pd.DataFrame(X_scaled, columns=selected_features)
            test_size = st.slider("Test set fraction (%)", 10, 40, 20, step=5)
            X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y, test_size=test_size/100, random_state=42)

            if st.button("🚀 Train Model"):
                with st.spinner("Training Random Forest..."):
                    time.sleep(0.25)
                    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42) if st.session_state["task_mode"] == "Classification" else RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    st.session_state["model"] = model
                    st.session_state["scaler"] = scaler
                    st.session_state["results"] = {
                        "task": st.session_state["task_mode"],
                        "y_test": y_test.tolist(),
                        "y_pred": np.array(y_pred).tolist(),
                        "model_name": f"Random Forest {st.session_state['task_mode']} Model",
                        "X_columns": selected_features,
                        "feature_importances": model.feature_importances_.tolist()
                    }
                    st.success("✅ Training completed. Go to 'Results' to inspect performance.")

            # Predict a new sample inline
            if st.session_state.get("model"):
                new_inputs = {}
                for f in selected_features:
                    new_inputs[f] = st.number_input(f"Value for {f}", value=float(np.median(df[f])) if f in df else 0.0, format="%.3f", key=f"input_{f}")
                if st.button("🔮 Predict Sample"):
                    input_df = pd.DataFrame([new_inputs])
                    input_scaled = st.session_state["scaler"].transform(input_df)
                    pred = st.session_state["model"].predict(input_scaled)
                    st.subheader("Prediction")
                    show_crop_suggestions(pred[0])

# ----------------- RESULTS -----------------
elif page == "📈 Results":
    st.title("📈 Model Results & Interpretation")

    if not st.session_state.get("results"):
        st.info("No trained model in session. Train a model first (Modeling or Quick Model).")
    else:
        results = st.session_state["results"]
        task = results["task"]
        y_test = np.array(results["y_test"])
        y_pred = np.array(results["y_pred"])

        st.subheader("Model Summary")
        st.write(f"**Model:** {results.get('model_name','Random Forest')}")
        st.write(f"**Features:** {', '.join(results.get('X_columns',[]))}")

        st.subheader("Performance Metrics")
        if task == "Classification":
            acc = accuracy_score(y_test, y_pred)
            st.metric("Accuracy", f"{acc:.3f}")
            cm = confusion_matrix(y_test, y_pred)
            st.write("Confusion Matrix:")
            st.write(cm)
        else:
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            st.write(f"RMSE: {rmse:.3f}, MAE: {mae:.3f}, R²: {r2:.3f}")
            st.subheader("Actual vs Predicted")
            fig = px.scatter(x=y_test, y=y_pred, labels={"x": "Actual", "y": "Predicted"}, title="Actual vs Predicted")
            st.plotly_chart(fig)

        st.subheader("Feature Importance")
        fi = results.get("feature_importances", [])
        feat = results.get("X_columns", [])
        if fi and feat:
            df_fi = pd.DataFrame({"Feature": feat, "Importance": fi}).sort_values("Importance", ascending=True)
            fig_fi = px.bar(df_fi, x="Importance", y="Feature", orientation="h", title="Feature Importance (Random Forest)")
            st.plotly_chart(fig_fi)

# ----------------- INSIGHTS -----------------
elif page == "🌿 Insights":
    st.title("🌿 Soil Health Insights")
    st.markdown("Automated soil health recommendations based on model outputs and feature signals.")
    if st.session_state["results"] is None:
        st.info("Train a model to get data-driven insights.")
    else:
        df = st.session_state["df"]
        fi = st.session_state["results"].get("feature_importances", [])
        feat = st.session_state["results"].get("X_columns", [])
        if fi and feat:
            df_fi = pd.DataFrame({"feature": feat, "importance": fi}).sort_values("importance", ascending=False)
            st.subheader("Top Features Influencing Predictions")
            for i, row in df_fi.head(5).iterrows():
                st.write(f"- **{row['feature']}** (importance: {row['importance']:.3f}) — median: {df[row['feature']].median():.3f}")
            st.markdown("### Recommendations")
            st.write("The suggestions below are generic. Use domain expertise before applying changes to soils.")
            if 'Nitrogen' in df.columns:
                median_n = df['Nitrogen'].median()
                if median_n < df['Nitrogen'].quantile(0.33):
                    st.info("Nitrogen tends to be low — consider organic amendments like compost or legume cover crops.")
                elif median_n > df['Nitrogen'].quantile(0.66):
                    st.warning("Nitrogen tends to be high — evaluate fertilizer application and leaching risks.")
                else:
                    st.success("Nitrogen levels are moderate across samples.")
        else:
            st.info("No model-based insights available yet. Train a model first.")

# ----------------- ABOUT -----------------
elif page == "👤 About":
    st.title("👤 About the Makers")
    st.markdown("Developed by:")
    st.write("")  # spacing
    col_a, col_b = st.columns([1,1])
    with col_a:
        render_profile("Andre Oneal A. Plaza", "profile_andre", "upload_andre")
    with col_b:
        render_profile("Rica Baliling", "profile_rica", "upload_rica")

    st.markdown("---")
    st.markdown("All thanks to God.")
    st.write("Developed for a capstone project.")
