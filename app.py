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

st.set_page_config(
    page_title="Machine Learning-Driven Soil Analysis for Sustainable Agriculture System",
    layout="wide",
    page_icon="🌿"
)

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
    "background_main": "linear-gradient(120deg, #fff0f6 0%, #ffe4ec 40%, #ffeef8 100%)",
    "sidebar_bg": "rgba(255, 239, 245, 0.95)",
    "primary_color": "#f06292",
    "secondary_color": "#f8bbd0",
    "button_gradient": "linear-gradient(90deg, #ff8aa2, #ff3b70)",
    "button_text": "#2e1c27",
    "header_glow_color_1": "#f8bbd0",
    "header_glow_color_2": "#f06292",
    "menu_icon_color": "#f06292",
    "nav_link_color": "#4a2a3a",
    "nav_link_selected_bg": "#ff93b0",
    "info_bg": "#fff1f6",
    "info_border": "#f06292",
    "success_bg": "#ffeef4",
    "success_border": "#f06292",
    "warning_bg": "#fff4e6",
    "warning_border": "#ffd580",
    "error_bg": "#fff0f0",
    "error_border": "#ffb3b3",
    "text_color": "#3a2a31",
    "title_color": "#70263b",
}

# Session state defaults
if "current_theme" not in st.session_state:
    st.session_state["current_theme"] = theme_classification
if "df" not in st.session_state:
    st.session_state["df"] = None
if "results" not in st.session_state:
    st.session_state["results"] = None
if "model" not in st.session_state:
    st.session_state["model"] = None
if "scaler" not in st.session_state:
    st.session_state["scaler"] = None
if "task_mode" not in st.session_state:
    st.session_state["task_mode"] = "Classification"  # default
if "trained_on_features" not in st.session_state:
    st.session_state["trained_on_features"] = None

# ----------------- THEME APPLIER -----------------
def apply_theme(theme):
    css = f"""
    <link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;600;700&family=Playfair+Display:wght@400;700&display=swap" rel="stylesheet">
    <style>
    .stApp {{ font-family: 'Montserrat', sans-serif; color:{theme['text_color']}; min-height:100vh; background: {theme['background_main']}; background-attachment: fixed; }}
    section[data-testid="stSidebar"] {{ background: {theme['sidebar_bg']} !important; border-radius:12px; padding:18px; box-shadow: 0 12px 40px rgba(0,0,0,0.06); }}
    div[data-testid="stSidebarNav"] a {{ color:{theme['nav_link_color']} !important; border-radius:8px; padding:10px; }}
    div[data-testid="stSidebarNav"] a:hover {{ background: rgba(0,0,0,0.03) !important; transform: translateX(4px); transition: all .12s; }}
    h1,h2,h3,h4,h5,h6 {{ color:{theme['title_color']}; font-family: 'Playfair Display', serif; }}
    .stButton button {{ background: {theme['button_gradient']} !important; color:{theme['button_text']} !important; border-radius:10px; padding:0.45rem 0.9rem; font-weight:700; }}
    .legend {{ background: rgba(0,0,0,0.04); border-radius:8px; padding:8px; color:{theme['text_color']}; font-size:13px; margin-top:10px; border: 1px solid {theme['primary_color']}30; }}
    .footer {{ text-align:center; color:{theme['secondary_color']}; font-size:13px; padding:10px; margin-top:18px; }}
    .stDownloadButton>button {{ background: {theme['button_gradient']} !important; color:{theme['button_text']} !important; }}
    /* Small form tweaks */
    div[data-testid="stToolbar"] {{ background: transparent; }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

apply_theme(st.session_state["current_theme"])

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.markdown("### 🌱 Soil Health System")
    st.markdown("Machine Learning-Driven Soil Analysis for Sustainable Agriculture System")
    st.write("---")
    selected = option_menu(
        None,
        ["🏠 Home", "📂 Upload Data", "📊 Visualization", "🤖 Modeling", "📈 Results", "🌿 Insights"],
        icons=["house", "cloud-upload", "bar-chart", "robot", "graph-up", "lightbulb"],
        menu_icon="list",
        default_index=1,
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
            "icon": {"color": st.session_state["current_theme"]["menu_icon_color"], "font-size": "18px"},
            "nav-link": {"font-size": "15px", "text-align": "left", "margin": "0px"},
            "nav-link-selected": {"background-color": st.session_state["current_theme"]["nav_link_selected_bg"]},
        }
    )
    st.write("---")
    st.markdown("Developed for sustainable agriculture — upload soil tests, visualize, and model.")
    st.caption("Tip: After upload use the quick 'Start Model' to get immediate Random Forest insights, or go to Modeling for advanced configuration.")

# ----------------- COMMON SETTINGS -----------------
column_mapping = {
    'pH': ['pH', 'ph', 'Soil_pH'],
    'Nitrogen': ['Nitrogen', 'N', 'Nitrogen_Level'],
    'Phosphorus': ['Phosphorus', 'P'],
    'Potassium': ['Potassium', 'K'],
    'Moisture': ['Moisture', 'Soil_Moisture'],
    'Organic Matter': ['Organic Matter', 'OM', 'oc']
}
required_columns = list(column_mapping.keys())

def safe_to_numeric_columns(df, cols):
    numeric_found = []
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            numeric_found.append(c)
    return numeric_found

def download_df_button(df, filename="final_preprocessed_soil_dataset.csv", label="⬇️ Download Cleaned & Preprocessed Data"):
    buf = BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    st.download_button(label=label, data=buf, file_name=filename, mime="text/csv")

def create_fertility_label(df, col="Nitrogen", q=3):
    labels = ['Low', 'Moderate', 'High']
    try:
        fert = pd.qcut(df[col], q=q, labels=labels, duplicates='drop')
        if fert.nunique() < 3:
            fert = pd.cut(df[col], bins=3, labels=labels)
    except Exception:
        fert = pd.cut(df[col], bins=3, labels=labels, include_lowest=True)
    return fert.astype(str)

def interpret_label(label):
    l = str(label).lower()
    if l in ["high", "good", "healthy", "3", "2.0"]:
        return ("Good", "green", "✅ Nutrients are balanced. Ideal for most crops.")
    if l in ["moderate", "medium", "2", "1.0"]:
        return ("Moderate", "orange", "⚠️ Some nutrient imbalance. Consider minor adjustments.")
    return ("Poor", "red", "🚫 Deficient or problematic — take corrective action.")

# ----------------- HOME -----------------
if selected == "🏠 Home":
    st.title("Machine Learning-Driven Soil Analysis for Sustainable Agriculture System")
    st.subheader("Welcome")
    st.markdown(
        """
        This interactive app helps agronomists, researchers, and farmers:
        - Upload & clean soil laboratory datasets.
        - Visualize distributions, correlations, and outliers.
        - Train Random Forest models (Classification for fertility bands or Regression for precise Nitrogen level).
        - Get model metrics, feature importance and make predictions for new soil samples.
        
        Quick workflow:
        1. Upload soil datasets in the 'Upload Data' tab.
        2. Use 'Start Model' for a quick Random Forest run or go to 'Modeling' for advanced control.
        3. Inspect results in 'Results' and iterate.
        """
    )
    st.info("Default modeling mode is Classification. You can switch to Regression (Sakura theme) in Upload Data or Modeling pages.")

# ----------------- UPLOAD DATA -----------------
elif selected == "📂 Upload Data":
    st.title("📂 Upload Soil Data")
    st.markdown("Upload one or more soil analysis files (.csv or .xlsx). The app will attempt to standardize column names and auto-preprocess numeric columns.")
    uploaded_files = st.file_uploader("Select datasets", type=['csv', 'xlsx'], accept_multiple_files=True)

    if st.session_state["df"] is not None and not uploaded_files:
        st.success(f"✅ Loaded preprocessed dataset ({st.session_state['df'].shape[0]} rows, {st.session_state['df'].shape[1]} cols).")
        st.dataframe(st.session_state["df"].head())
        if st.button("🔁 Clear current dataset"):
            st.session_state["df"] = None
            st.session_state["results"] = None
            st.session_state["model"] = None
            st.session_state["scaler"] = None
            st.experimental_rerun()

    cleaned_dfs = []
    if uploaded_files:
        for file in uploaded_files:
            try:
                df_file = pd.read_csv(file) if file.name.endswith('.csv') else pd.read_excel(file)
                renamed = {}
                for std_col, alt_names in column_mapping.items():
                    for alt in alt_names:
                        if alt in df_file.columns:
                            renamed[alt] = std_col
                            break
                df_file.rename(columns=renamed, inplace=True)
                cols_to_keep = [col for col in required_columns if col in df_file.columns]
                df_file = df_file[cols_to_keep]
                safe_to_numeric_columns(df_file, cols_to_keep)
                df_file.drop_duplicates(inplace=True)
                cleaned_dfs.append(df_file)
                st.success(f"✅ Cleaned {file.name} — kept: {', '.join(cols_to_keep)} ({df_file.shape[0]} rows)")
            except Exception as e:
                st.warning(f"⚠️ Skipped {file.name}: {e}")

        if cleaned_dfs:
            df = pd.concat(cleaned_dfs, ignore_index=True, sort=False)
            df.replace(r'^\s*$', np.nan, regex=True, inplace=True)
            safe_to_numeric_columns(df, required_columns)

            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                medians = df[numeric_cols].median()
                df[numeric_cols] = df[numeric_cols].fillna(medians)

            cat_cols = df.select_dtypes(exclude=[np.number]).columns
            for c in cat_cols:
                try:
                    if df[c].isnull().sum() > 0:
                        df[c].fillna(df[c].mode().iloc[0], inplace=True)
                except Exception:
                    df[c].fillna(method='ffill', inplace=True)

            df.dropna(how='all', inplace=True)
            st.session_state["df"] = df
            st.success("✨ Dataset preprocessed and stored in session.")
            st.write(f"Rows: {df.shape[0]} — Columns: {df.shape[1]}")
            st.dataframe(df.head())
            download_df_button(df)

            # Quick mode selector and Start Model button (per user request)
            st.markdown("### Quick Modeling (one-click)")
            col1, col2 = st.columns([2, 1])
            with col1:
                quick_mode = st.radio("Select quick modeling mode:", ["Classification", "Regression"],
                                      index=0, key="quick_mode_radio")
            with col2:
                if quick_mode != st.session_state["task_mode"]:
                    # change theme appropriately
                    st.session_state["task_mode"] = quick_mode
                    st.session_state["current_theme"] = theme_classification if quick_mode == "Classification" else theme_sakura
                    apply_theme(st.session_state["current_theme"])

                if st.button("▶️ Start Quick Model"):
                    # Validate presence of Nitrogen
                    if 'Nitrogen' not in df.columns:
                        st.error("The 'Nitrogen' column is required for modeling. Please ensure your uploaded data includes Nitrogen.")
                    else:
                        # Prepare X, y
                        df_local = df.copy()
                        if quick_mode == "Classification":
                            df_local['Fertility_Level'] = create_fertility_label(df_local, col='Nitrogen', q=3)
                            y = df_local['Fertility_Level']
                        else:
                            y = df_local['Nitrogen']

                        X = df_local.select_dtypes(include=[np.number]).drop(columns=['Nitrogen'], errors='ignore')
                        if X.shape[1] == 0:
                            st.error("No numeric features available for quick modeling after dropping the target.")
                        else:
                            # Quick scaler, split, train
                            scaler = MinMaxScaler()
                            X_scaled = scaler.fit_transform(X)
                            X_df = pd.DataFrame(X_scaled, columns=X.columns)
                            X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=0.2, random_state=42)

                            if quick_mode == "Classification":
                                model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
                            else:
                                model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)

                            with st.spinner("Training quick Random Forest..."):
                                model.fit(X_train, y_train)
                                y_pred = model.predict(X_test)

                            # Save into session
                            st.session_state["model"] = model
                            st.session_state["scaler"] = scaler
                            st.session_state["results"] = {
                                "task": quick_mode,
                                "y_test": y_test.tolist(),
                                "y_pred": np.array(y_pred).tolist(),
                                "model_name": "Random Forest (Quick)",
                                "X_columns": X.columns.tolist(),
                                "feature_importances": model.feature_importances_.tolist()
                            }
                            st.session_state["trained_on_features"] = X.columns.tolist()
                            st.success("✅ Quick model trained. Navigate to 'Results' to inspect performance.")
        else:
            st.error("No valid sheets processed. Check file formats and column headers.")

# ----------------- VISUALIZATION -----------------
elif selected == "📊 Visualization":
    st.title("📊 Data Visualization")
    st.markdown("Explore distributions, correlations, and relationships in your preprocessed data.")
    if st.session_state["df"] is None:
        st.info("Please upload data first in 'Upload Data'.")
    else:
        df = st.session_state["df"]
        if 'Nitrogen' in df.columns and 'Fertility_Level' not in df.columns:
            df['Fertility_Level'] = create_fertility_label(df, col='Nitrogen', q=3)

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            st.warning("No numeric columns available for visualization.")
        else:
            st.subheader("Distributions")
            col1, col2 = st.columns(2)
            with col1:
                dist_feature = st.selectbox("Distribution feature", numeric_cols, key="vis_dist_feature")
                bins = st.slider("Bins", 10, 80, 30, key="vis_bins")
            with col2:
                scatter_x = st.selectbox("Scatter X", numeric_cols, index=0, key="vis_scatter_x")
                scatter_y = st.selectbox("Scatter Y", numeric_cols, index=1 if len(numeric_cols) > 1 else 0, key="vis_scatter_y")

            fig_hist = px.histogram(df, x=dist_feature, nbins=bins, marginal="box",
                                    title=f"Distribution of {dist_feature}",
                                    color_discrete_sequence=[st.session_state["current_theme"]["primary_color"]])
            fig_hist.update_layout(template="plotly_dark")
            st.plotly_chart(fig_hist, use_container_width=True)

            st.subheader("Correlation Matrix")
            corr = df[numeric_cols].corr()
            fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale=px.colors.sequential.Viridis, title="Correlation Heatmap")
            fig_corr.update_layout(template="plotly_dark")
            st.plotly_chart(fig_corr, use_container_width=True)

            st.subheader("Scatter with Fertility Level")
            fig_scatter = px.scatter(df, x=scatter_x, y=scatter_y,
                                     color='Fertility_Level' if 'Fertility_Level' in df.columns else None,
                                     color_discrete_map={'Low': 'red', 'Moderate': 'orange', 'High': 'green'},
                                     hover_data=df.columns, title=f"{scatter_x} vs {scatter_y}")
            fig_scatter.update_layout(template="plotly_dark")
            st.plotly_chart(fig_scatter, use_container_width=True)

# ----------------- MODELING -----------------
elif selected == "🤖 Modeling":
    st.title("🤖 Modeling — Random Forest")
    st.markdown("Fine tune hyperparameters and train Random Forest models for Classification or Regression.")

    if st.session_state["df"] is None:
        st.info("Please upload a dataset first in 'Upload Data'.")
    else:
        df = st.session_state["df"].copy()
        # task selection and theme switch
        st.subheader("Select Task")
        task = st.radio("Choose modeling task:", ["Classification", "Regression"], index=0 if st.session_state["task_mode"] == "Classification" else 1)
        if task != st.session_state["task_mode"]:
            st.session_state["task_mode"] = task
            st.session_state["current_theme"] = theme_classification if task == "Classification" else theme_sakura
            apply_theme(st.session_state["current_theme"])

        st.markdown(f"Current Mode: **{st.session_state['task_mode']}**", unsafe_allow_html=True)

        if 'Nitrogen' not in df.columns:
            st.error("Missing 'Nitrogen' column required as target. Ensure your dataset contains 'Nitrogen'.")
            st.stop()

        # prepare target and features
        if st.session_state["task_mode"] == "Classification":
            df['Fertility_Level'] = create_fertility_label(df, col='Nitrogen', q=3)
            y = df['Fertility_Level']
        else:
            y = df['Nitrogen']

        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        # drop Nitrogen from features
        if 'Nitrogen' in numeric_features:
            numeric_features.remove('Nitrogen')

        st.subheader("Feature Selection")
        st.markdown("Select numeric features to include in the model.")
        selected_features = st.multiselect("Features", options=numeric_features, default=numeric_features)

        if not selected_features:
            st.warning("Select at least one feature.")
            st.stop()

        X = df[selected_features]

        st.subheader("Hyperparameters")
        col1, col2 = st.columns(2)
        with col1:
            n_estimators = st.slider("n_estimators", 50, 500, 150, step=50)
        with col2:
            max_depth = st.slider("max_depth", 2, 50, 12)

        # scaling
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=selected_features)

        # splitting
        test_size = st.slider("Test set fraction", 10, 40, 20, step=5)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y, test_size=test_size/100, random_state=42)

        if st.button("🚀 Train Model"):
            with st.spinner("Training Random Forest..."):
                time.sleep(0.3)
                if st.session_state["task_mode"] == "Classification":
                    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1)
                else:
                    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1)

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                # cross-validation summary
                try:
                    cv_scores = cross_val_score(model, X_scaled_df, y, cv=5, scoring='accuracy' if st.session_state["task_mode"] == "Classification" else 'r2')
                    cv_summary = {"mean_cv": float(np.mean(cv_scores)), "std_cv": float(np.std(cv_scores))}
                except Exception:
                    cv_summary = None

                st.session_state["model"] = model
                st.session_state["scaler"] = scaler
                st.session_state["results"] = {
                    "task": st.session_state["task_mode"],
                    "y_test": y_test.tolist(),
                    "y_pred": np.array(y_pred).tolist(),
                    "model_name": "Random Forest",
                    "X_columns": selected_features,
                    "feature_importances": model.feature_importances_.tolist(),
                    "cv_summary": cv_summary
                }
                st.session_state["trained_on_features"] = selected_features
                st.success("✅ Training completed. Go to 'Results' to inspect performance.")

        st.markdown("### Predict a New Sample")
        if st.session_state.get("model"):
            new_inputs = {}
            for f in selected_features:
                new_inputs[f] = st.number_input(f"Value for {f}", value=float(np.median(df[f])) if f in df else 0.0, format="%.3f", key=f"input_{f}")
            if st.button("🔮 Predict Sample"):
                input_df = pd.DataFrame([new_inputs])
                scaler_local = st.session_state["scaler"] if st.session_state.get("scaler") else MinMaxScaler().fit(df[selected_features])
                input_scaled = scaler_local.transform(input_df)
                pred = st.session_state["model"].predict(input_scaled)
                st.subheader("Prediction")
                if st.session_state["task_mode"] == "Classification":
                    pred_label, color, expl = interpret_label(pred[0])
                    st.markdown(f"**Predicted Fertility:** <span style='color:{color};font-weight:700'>{pred_label}</span>", unsafe_allow_html=True)
                    st.write(expl)
                else:
                    st.markdown(f"**Predicted Nitrogen:** <span style='color:{st.session_state['current_theme']['primary_color']};font-weight:700'>{pred[0]:.3f}</span>", unsafe_allow_html=True)
        else:
            st.info("Train a model in this page or use the 'Start Quick Model' from Upload Data to get a quick result.")

# ----------------- RESULTS -----------------
elif selected == "📈 Results":
    st.title("📈 Model Results & Interpretation")
    if not st.session_state.get("results"):
        st.info("No trained model in session. Train a model first (Modeling or Quick Model).")
    else:
        results = st.session_state["results"]
        task = results["task"]
        y_test = np.array(results["y_test"])
        y_pred = np.array(results["y_pred"])

        st.subheader("Model Summary")
        st.write(f"Model: **{results.get('model_name','Random Forest')}**")
        st.write(f"Task: **{task}**")
        if results.get("cv_summary"):
            cv = results["cv_summary"]
            st.write(f"Cross-validation mean score: **{cv['mean_cv']:.3f}** (std: {cv['std_cv']:.3f})")

        st.subheader("Performance Metrics")
        if task == "Classification":
            acc = accuracy_score(y_test, y_pred)
            st.metric("Accuracy", f"{acc:.3f}")
            st.markdown("**Classification Report**")
            try:
                report = classification_report(y_test, y_pred, output_dict=False)
                st.text(report)
            except Exception:
                st.text(classification_report(y_test, y_pred))
            cm = confusion_matrix(y_test, y_pred, labels=['Low', 'Moderate', 'High'])
            fig_cm = px.imshow(cm, text_auto=True, title="Confusion Matrix (Low / Moderate / High)")
            fig_cm.update_layout(template="plotly_dark")
            st.plotly_chart(fig_cm, use_container_width=True)
        else:
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            st.metric("RMSE", f"{rmse:.3f}")
            st.metric("MAE", f"{mae:.3f}")
            st.metric("R²", f"{r2:.3f}")

            # Scatter y_test vs y_pred
            df_res = pd.DataFrame({"y_test": y_test, "y_pred": y_pred})
            fig_scatter = px.scatter(df_res, x="y_test", y="y_pred", trendline="ols",
                                     title="Actual vs Predicted Nitrogen")
            fig_scatter.update_layout(template="plotly_dark")
            st.plotly_chart(fig_scatter, use_container_width=True)

            # Residuals
            df_res["residual"] = df_res["y_test"] - df_res["y_pred"]
            fig_res = px.histogram(df_res, x="residual", nbins=30, title="Residual Distribution")
            fig_res.update_layout(template="plotly_dark")
            st.plotly_chart(fig_res, use_container_width=True)

        st.subheader("Feature Importances")
        fi = results.get("feature_importances", [])
        feat = results.get("X_columns", [])
        if fi and feat:
            df_fi = pd.DataFrame({"feature": feat, "importance": fi}).sort_values("importance", ascending=False)
            fig_fi = px.bar(df_fi, x="importance", y="feature", orientation="h", title="Feature Importances")
            fig_fi.update_layout(template="plotly_dark")
            st.plotly_chart(fig_fi, use_container_width=True)
        else:
            st.info("No feature importances available.")

        st.markdown("---")
        st.markdown("You can save the trained model and scaler for later use:")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 Save Model"):
                if st.session_state.get("model"):
                    joblib.dump(st.session_state["model"], "rf_model.joblib")
                    st.success("Model saved as rf_model.joblib")
                else:
                    st.warning("No model in session to save.")
        with col2:
            if st.button("💾 Save Scaler"):
                if st.session_state.get("scaler"):
                    joblib.dump(st.session_state["scaler"], "scaler.joblib")
                    st.success("Scaler saved as scaler.joblib")
                else:
                    st.warning("No scaler in session to save.")

# ----------------- INSIGHTS -----------------
elif selected == "🌿 Insights":
    st.title("🌿 Soil Health Insights")
    st.markdown("Automated soil health recommendations based on model outputs and feature signals.")
    if st.session_state["results"] is None:
        st.info("Train a model to get data-driven insights.")
    else:
        # Basic insights derived from trained model's feature importances and sample medians
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
