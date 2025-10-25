# (Full file begins)
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
# profile images stored in session as base64
if "profile_andre" not in st.session_state:
    st.session_state["profile_andre"] = None
if "profile_rica" not in st.session_state:
    st.session_state["profile_rica"] = None
# navigation index for sidebar
if "page_index" not in st.session_state:
    st.session_state["page_index"] = 0

# ----------------- THEME APPLIER + BACKGROUND -----------------
def apply_theme(theme):
    css = f"""
    <link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;600;700&family=Playfair+Display:wght@400;700&display=swap" rel="stylesheet">
    <style>
    /* Main app + background decorative shapes */
    .stApp {{
      font-family: 'Montserrat', sans-serif;
      color:{theme['text_color']};
      min-height:100vh;
      background: {theme['background_main']};
      background-attachment: fixed;
      position: relative;
      overflow: hidden;
    }}
    /* soft decorative blobs in background */
    .bg-decor {{
      position: absolute;
      right: -8%;
      top: -12%;
      width: 55vmax;
      height: 55vmax;
      background: radial-gradient(circle at 20% 20%, rgba(255,255,255,0.03), transparent 10%),
                  radial-gradient(circle at 80% 80%, rgba(255,255,255,0.02), transparent 25%);
      transform: rotate(12deg) scale(1.1);
      filter: blur(36px);
      z-index: 0;
      pointer-events: none;
    }}
    section[data-testid="stSidebar"] {{
      background: {theme['sidebar_bg']} !important;
      border-radius:12px;
      padding:18px;
      box-shadow: 0 12px 40px rgba(0,0,0,0.12);
      z-index: 2;
    }}
    /* modernized header card in sidebar */
    .sidebar-header {{
      padding: 12px;
      border-radius: 10px;
      background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
      margin-bottom: 12px;
      border: 1px solid rgba(255,255,255,0.03);
    }}
    .sidebar-title {{ font-family: 'Playfair Display', serif; color:{theme['title_color']}; margin:0; }}
    .sidebar-sub {{ font-size:12px; color:{theme['secondary_color']}; margin-top:6px; opacity:0.95; }}
    /* make the menu links a bit more spaced and larger hit area */
    div[data-testid="stSidebarNav"] a {{
      color:{theme['nav_link_color']} !important;
      border-radius:8px;
      padding:10px 12px!important;
      margin-bottom:6px!important;
      display:block!important;
      transition: all .12s;
      font-size:15px!important;
    }}
    div[data-testid="stSidebarNav"] a:hover {{ background: rgba(255,255,255,0.03) !important; transform: translateX(6px); }}
    div[data-testid="stSidebarNav"] a[aria-current="page"] {{ background:{theme['nav_link_selected_bg']}!important; color:#0b0b0b!important; box-shadow:0 6px 16px rgba(0,0,0,0.25); }}
    h1,h2,h3,h4,h5,h6 {{ color:{theme['title_color']}; font-family: 'Playfair Display', serif; z-index: 3; }}
    .stButton button {{ background: {theme['button_gradient']} !important; color:{theme['button_text']} !important; border-radius:10px; padding:0.45rem 0.9rem; font-weight:700; }}
    .stDownloadButton>button {{ background: {theme['button_gradient']} !important; color:{theme['button_text']} !important; }}
    .profile-circle {{
      width:120px;
      height:120px;
      border-radius:50%;
      display:inline-block;
      vertical-align: middle;
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.04), 0 8px 18px rgba(0,0,0,0.35);
      background: linear-gradient(135deg, rgba(255,255,255,0.03), rgba(0,0,0,0.04));
      border: 4px solid rgba(255,255,255,0.04);
      overflow:hidden;
      text-align:center;
      line-height:120px;
    }}
    .profile-holo {{
      background: linear-gradient(135deg, rgba(255,255,255,0.02), rgba(255,255,255,0.00));
      padding:6px;
      border-radius:50%;
      display:inline-block;
    }}
    .profile-name {{ margin-top:8px; font-weight:700; color:{theme['secondary_color']}; }}
    .uploader-hint {{ font-size:12px; color:{theme['text_color']}; opacity:0.7; }}
    /* small form tweaks */
    div[data-testid="stToolbar"] {{ background: transparent; }}
    /* reduce verbose text in option_menu button (we removed instructions by default) */
    .card {{
      background: rgba(255,255,255,0.02);
      padding: 12px;
      border-radius: 10px;
      border:1px solid rgba(255,255,255,0.03);
      margin-bottom:8px;
    }}
    .metric-desc {{ font-size:12px; color:rgba(255,255,255,0.85); opacity:0.9; }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)
    # decorative blobs (single element)
    st.markdown('<div class="bg-decor"></div>', unsafe_allow_html=True)

apply_theme(st.session_state["current_theme"])

# ----------------- SIDEBAR (redesigned) -----------------
# New order: Home, Modeling, Visualization, Results, Insights, About
PAGES = ["🏠 Home", "🤖 Modeling", "📊 Visualization", "📈 Results", "🌿 Insights", "👤 About"]

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

    default_index = st.session_state.get("page_index", 0)
    selected = option_menu(
        None,
        PAGES,
        icons=["house", "robot", "bar-chart", "graph-up", "lightbulb", "person-circle"],
        menu_icon="list",
        default_index=default_index,
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
            "icon": {"color": st.session_state["current_theme"]["menu_icon_color"], "font-size": "18px"},
            "nav-link": {"font-size": "15px", "text-align": "left", "margin": "0px"},
            "nav-link-selected": {"background-color": st.session_state["current_theme"]["nav_link_selected_bg"]},
        }
    )
    # persist index for programmatic navigation
    try:
        st.session_state["page_index"] = PAGES.index(selected)
    except Exception:
        st.session_state["page_index"] = 0

    st.write("---")
    # minimal footer text
    st.markdown(f"<div style='font-size:12px;color:{st.session_state['current_theme']['text_color']};opacity:0.85'>Developed for sustainable agriculture</div>", unsafe_allow_html=True)

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

# ----------------- PROFILE / AVATAR HELPERS -----------------
def pil_to_base64(img: Image.Image, fmt="PNG"):
    buf = sysio.BytesIO()
    img.save(buf, format=fmt)
    b = base64.b64encode(buf.getvalue()).decode("utf-8")
    return b

def render_profile(name, session_key, upload_key):
    """
    REPLACED function: now uses static images from assets/ folder (andre.png / rica.png)
    and displays a neon-glow circular avatar. Uploaders and hints removed.
    Signature kept the same to avoid touching other code that calls this function.
    """
    # container
    st.markdown("<div style='display:flex;flex-direction:column;align-items:center;text-align:center;'>", unsafe_allow_html=True)

    # determine filename by name
    image_filename = None
    if "Andre" in name:
        image_filename = "andre.png"
    elif "Rica" in name:
        image_filename = "rica.png"

    img_b64 = None
    if image_filename:
        image_path = os.path.join("assets", image_filename)
        if os.path.exists(image_path):
            try:
                with open(image_path, "rb") as f:
                    img_b64 = base64.b64encode(f.read()).decode("utf-8")
            except Exception:
                img_b64 = None

    # neon glow CSS (keeps visual style but isolated here)
    st.markdown("""
    <style>
    .neon-glow {
        width:132px;
        height:132px;
        border-radius:50%;
        padding:6px;
        display:inline-flex;
        align-items:center;
        justify-content:center;
        box-shadow: 0 0 20px rgba(129,199,132,0.35), 0 0 40px rgba(165,214,167,0.18);
        background: radial-gradient(circle at 50% 50%, rgba(129,199,132,0.12), rgba(0,0,0,0.00));
    }
    .neon-img {
        width:120px;
        height:120px;
        border-radius:50%;
        object-fit:cover;
        border:3px solid rgba(255,255,255,0.06);
        display:block;
    }
    </style>
    """, unsafe_allow_html=True)

    if img_b64:
        html = f"""
        <div class="neon-glow">
            <img class="neon-img" src="data:image/png;base64,{img_b64}" />
        </div>
        """
    else:
        html = """
        <div class="neon-glow">
            <div style="font-size:44px;opacity:0.75;">👤</div>
        </div>
        """

    st.markdown(html, unsafe_allow_html=True)

    # name + centered BSIS 4-A (replaces uploader hint + "Upload an image..." text)
    st.markdown(f"<div style='margin-top:8px;font-weight:700;color:{st.session_state['current_theme']['secondary_color']};'>{name}</div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:14px;color:rgba(255,255,255,0.85);margin-top:4px;font-weight:600;'>BSIS 4-A</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ----------------- Reusable upload & preprocess function (used on Home) -----------------
def upload_and_preprocess_widget():
    st.markdown("### 📂 Upload Soil Data")
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
        else:
            st.error("No valid sheets processed. Check file formats and column headers.")

# ----------------- HOME -----------------
if selected == "🏠 Home":
    st.title("Machine Learning-Driven Soil Analysis for Sustainable Agriculture System")
    st.markdown("<small style='color:rgba(255,255,255,0.75)'>Capstone Project</small>", unsafe_allow_html=True)
    st.write("---")

    # No task-mode toggle on Home (moved to Modeling). Home only handles upload + navigation.
    upload_and_preprocess_widget()

    # After upload, give clear navigation buttons to proceed
    if st.session_state.get("df") is not None:
        st.write("---")
        st.markdown("Proceed to:")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("➡️ Proceed to Modeling"):
                st.session_state["page_index"] = PAGES.index("🤖 Modeling")
                st.experimental_rerun()
        with col2:
            if st.button("➡️ Proceed to Visualization"):
                st.session_state["page_index"] = PAGES.index("📊 Visualization")
                st.experimental_rerun()

# ----------------- MODELING -----------------
elif selected == "🤖 Modeling":
    st.title("🤖 Modeling — Random Forest")
    st.markdown("Fine tune hyperparameters and train Random Forest models for Classification or Regression.")

    if st.session_state["df"] is None:
        st.info("Please upload a dataset first in 'Home'.")
    else:
        df = st.session_state["df"].copy()
        st.subheader("Select Task")
        # Keep task radio here (single place to change mode)
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
                    "model_name": f"Random Forest {st.session_state['task_mode']} Model",
                    "X_columns": selected_features,
                    "feature_importances": model.feature_importances_.tolist(),
                    "cv_summary": cv_summary,
                    "n_estimators": n_estimators,
                    "max_depth": max_depth,
                    "train_size": X_train.shape[0],
                    "test_size": X_test.shape[0]
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
            st.info("Train a model in this page to enable predictions. Modeling is the canonical place to run/train models.")

# ----------------- VISUALIZATION -----------------
elif selected == "📊 Visualization":
    st.title("📊 Data Visualization")
    st.markdown("Explore distributions, correlations, and relationships in your preprocessed data.")
    if st.session_state["df"] is None:
        st.info("Please upload data first in 'Home' (Upload Data is integrated there).")
    else:
        df = st.session_state["df"].copy()
        # Ensure fertility label for classification visualizations exists if needed
        if 'Nitrogen' in df.columns and 'Fertility_Level' not in df.columns:
            df['Fertility_Level'] = create_fertility_label(df, col='Nitrogen', q=3)

        # Parameter overview charts (histograms + brief explanation)
        st.markdown("### Parameter Overview")
        cols_for_overview = ['pH', 'Nitrogen', 'Phosphorus', 'Potassium', 'Moisture', 'Organic Matter']
        present = [c for c in cols_for_overview if c in df.columns]

        # layout: two columns for histograms
        for i in range(0, len(present), 2):
            c1 = present[i]
            c2 = present[i+1] if i+1 < len(present) else None
            col1, col2 = st.columns(2)
            with col1:
                feature = c1
                fig = px.histogram(df, x=feature, nbins=30, title=f"Distribution of {feature}")
                fig.update_layout(template="plotly_dark", height=320)
                st.plotly_chart(fig, use_container_width=True)
                # explanation
                if feature == 'pH':
                    st.markdown("<div class='metric-desc'>Shows whether soil samples are acidic, neutral, or alkaline — important for nutrient availability.</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='metric-desc'>Distribution of {feature} across samples — helps detect skew, outliers, and typical levels.</div>", unsafe_allow_html=True)
            with col2:
                if c2:
                    feature = c2
                    fig = px.histogram(df, x=feature, nbins=30, title=f"Distribution of {feature}")
                    fig.update_layout(template="plotly_dark", height=320)
                    st.plotly_chart(fig, use_container_width=True)
                    st.markdown(f"<div class='metric-desc'>Distribution of {feature} across samples — helps detect skew, outliers, and typical levels.</div>", unsafe_allow_html=True)

        st.markdown("### Correlation Heatmap")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) >= 2:
            corr = df[numeric_cols].corr()
            fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale=px.colors.sequential.Viridis, title="Correlation Heatmap")
            fig_corr.update_layout(template="plotly_dark", height=480)
            st.plotly_chart(fig_corr, use_container_width=True)
            st.markdown("<div class='metric-desc'>Correlation matrix highlights relationships between soil properties (positive/negative correlations).</div>", unsafe_allow_html=True)
        else:
            st.info("Not enough numeric columns to generate a correlation heatmap.")

        st.write("---")
        st.markdown("### Mode-specific Visualizations")
        mode = st.session_state.get("task_mode", "Classification")
        st.markdown(f"Current Mode: **{mode}**. (Change mode in Modeling.)")

        if mode == "Classification":
            # Show fertility-level colored histograms / stacked distributions
            if 'Fertility_Level' not in df.columns:
                df['Fertility_Level'] = create_fertility_label(df, col='Nitrogen', q=3)
            st.markdown("#### Fertility Level Breakdown")
            col1, col2 = st.columns(2)
            with col1:
                if 'Nitrogen' in df.columns:
                    fig_n = px.histogram(df, x='Nitrogen', color='Fertility_Level', nbins=30, barmode='overlay', title="Nitrogen distribution by Fertility Level")
                    fig_n.update_layout(template="plotly_dark", height=380)
                    st.plotly_chart(fig_n, use_container_width=True)
                    st.markdown("<div class='metric-desc'>Shows how Nitrogen values map to the fertility labels (Low/Moderate/High).</div>", unsafe_allow_html=True)
                else:
                    st.info("Nitrogen column missing for fertility-level visualization.")
            with col2:
                # example: pH by fertility level if present
                if 'pH' in df.columns:
                    fig_ph = px.violin(df, x='Fertility_Level', y='pH', box=True, title="pH distribution per Fertility Level")
                    fig_ph.update_layout(template="plotly_dark", height=380)
                    st.plotly_chart(fig_ph, use_container_width=True)
                    st.markdown("<div class='metric-desc'>pH distribution across fertility classes — identifies pH shifts by class.</div>", unsafe_allow_html=True)
                else:
                    st.info("pH not available for a pH-by-class plot.")
        else:
            # Regression mode visuals (Actual vs Predicted etc.)
            st.markdown("#### Regression Visuals — Fertility Prediction (Nitrogen)")
            if st.session_state.get("results") and st.session_state["results"].get("task") == "Regression":
                results = st.session_state["results"]
                y_test = np.array(results["y_test"])
                y_pred = np.array(results["y_pred"])
                df_res = pd.DataFrame({"Actual_Nitrogen": y_test, "Predicted_Nitrogen": y_pred})
                col1, col2 = st.columns([1.2, 1])
                with col1:
                    # Actual vs Predicted with OLS trendline (requires statsmodels)
                    try:
                        fig1 = px.scatter(df_res, x="Actual_Nitrogen", y="Predicted_Nitrogen", trendline="ols",
                                          title="Actual vs Predicted Nitrogen (Model Predictions)")
                        fig1.update_layout(template="plotly_dark", height=420)
                        st.plotly_chart(fig1, use_container_width=True)
                        st.markdown("<div class='metric-desc'>Actual vs Predicted with a linear fit — closer to diagonal means better predictions.</div>", unsafe_allow_html=True)
                    except Exception as e:
                        # fallback
                        fig1 = px.scatter(df_res, x="Actual_Nitrogen", y="Predicted_Nitrogen", title="Actual vs Predicted Nitrogen (Model Predictions)")
                        fig1.update_layout(template="plotly_dark", height=420)
                        st.plotly_chart(fig1, use_container_width=True)
                        st.markdown("<div class='metric-desc'>Actual vs Predicted (trendline unavailable). Install statsmodels to enable OLS trendline.</div>", unsafe_allow_html=True)
                with col2:
                    # residuals
                    df_res["residual"] = df_res["Actual_Nitrogen"] - df_res["Predicted_Nitrogen"]
                    fig_res = px.histogram(df_res, x="residual", nbins=30, title="Residual Distribution (Actual - Predicted)")
                    fig_res.update_layout(template="plotly_dark", height=420)
                    st.plotly_chart(fig_res, use_container_width=True)
                    st.markdown("<div class='metric-desc'>Residual distribution shows bias & spread — ideally centered near zero and narrow.</div>", unsafe_allow_html=True)
            else:
                st.info("No regression model results available. Train a Regression model in Modeling first.")

# ----------------- RESULTS -----------------
elif selected == "📈 Results":
    st.title("📈 Model Results & Interpretation")
    if not st.session_state.get("results"):
        st.info("No trained model in session. Train a model first (Modeling).")
    else:
        results = st.session_state["results"]
        task = results["task"]
        y_test = np.array(results["y_test"])
        y_pred = np.array(results["y_pred"])

        # Model summary card
        st.markdown("<div class='card'><strong>Model Summary</strong></div>", unsafe_allow_html=True)
        col_a, col_b = st.columns([2,1])
        with col_a:
            st.write(f"**Model:** {results.get('model_name','Random Forest')}")
            if results.get("cv_summary"):
                cv = results["cv_summary"]
                st.write(f"Cross-val mean: **{cv['mean_cv']:.3f}** (std: {cv['std_cv']:.3f})")
            st.write(f"Trained features: **{', '.join(results.get('X_columns', []))}**")
            st.write(f"Training set size: **{results.get('train_size','-')}**, Test set size: **{results.get('test_size','-')}**")
        with col_b:
            st.markdown("<div class='card'>Hyperparameters</div>", unsafe_allow_html=True)
            st.write(f"- n_estimators: **{results.get('n_estimators','-')}**")
            st.write(f"- max_depth: **{results.get('max_depth','-')}**")

        st.markdown("---")
        st.subheader("Performance Metrics")
        if task == "Classification":
            # Compact metrics row
            col1, col2, col3 = st.columns(3)
            try:
                acc = accuracy_score(y_test, y_pred)
                col1.metric("Accuracy", f"{acc:.3f}")
                col1.markdown("<div class='metric-desc'>Fraction of correct predictions.</div>", unsafe_allow_html=True)
            except Exception:
                col1.write("Accuracy N/A")
            # show confusion matrix
            cm = confusion_matrix(y_test, y_pred, labels=['Low', 'Moderate', 'High'])
            fig_cm = px.imshow(cm, text_auto=True, title="Confusion Matrix (Low / Moderate / High)")
            fig_cm.update_layout(template="plotly_dark", height=360)
            st.plotly_chart(fig_cm, use_container_width=True)
            st.markdown("<div class='metric-desc'>Rows: actual classes. Columns: predicted classes. Diagonal = correct predictions.</div>", unsafe_allow_html=True)

            st.markdown("**Classification Report**")
            try:
                report = classification_report(y_test, y_pred, output_dict=False)
                st.text(report)
            except Exception:
                st.text(classification_report(y_test, y_pred))
        else:
            # Regression metrics explained clearly
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("RMSE", f"{rmse:.3f}")
            col1.markdown("<div class='metric-desc'>Root Mean Square Error — lower is better.</div>", unsafe_allow_html=True)
            col2.metric("MAE", f"{mae:.3f}")
            col2.markdown("<div class='metric-desc'>Mean Absolute Error — average absolute error.</div>", unsafe_allow_html=True)
            col3.metric("R²", f"{r2:.3f}")
            col3.markdown("<div class='metric-desc'>R²: variance explained by the model (1.0 is perfect).</div>", unsafe_allow_html=True)

            # scatter actual vs predicted (with trendline if available)
            df_res = pd.DataFrame({"Actual_Nitrogen": y_test, "Predicted_Nitrogen": y_pred})
            col1, col2 = st.columns([1.5, 1])
            with col1:
                try:
                    fig_sc = px.scatter(df_res, x="Actual_Nitrogen", y="Predicted_Nitrogen", trendline="ols", title="Actual vs Predicted Nitrogen")
                    fig_sc.update_layout(template="plotly_dark", height=420)
                    st.plotly_chart(fig_sc, use_container_width=True)
                    st.markdown("<div class='metric-desc'>Ideal model lies near the diagonal line; trendline shows overall bias.</div>", unsafe_allow_html=True)
                except Exception:
                    fig_sc = px.scatter(df_res, x="Actual_Nitrogen", y="Predicted_Nitrogen", title="Actual vs Predicted Nitrogen")
                    fig_sc.update_layout(template="plotly_dark", height=420)
                    st.plotly_chart(fig_sc, use_container_width=True)
                    st.markdown("<div class='metric-desc'>Install statsmodels to enable OLS trendlines.</div>", unsafe_allow_html=True)
            with col2:
                df_res["residual"] = df_res["Actual_Nitrogen"] - df_res["Predicted_Nitrogen"]
                fig_res = px.histogram(df_res, x="residual", nbins=30, title="Residual Distribution")
                fig_res.update_layout(template="plotly_dark", height=420)
                st.plotly_chart(fig_res, use_container_width=True)
                st.markdown("<div class='metric-desc'>Residuals should be centered around 0 for an unbiased model.</div>", unsafe_allow_html=True)

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

# ----------------- ABOUT / PROFILE -----------------
elif selected == "👤 About":
    st.title("👤 About the Makers")
    st.markdown("Developed by:")
    st.write("")  # spacing
    col_a, col_b = st.columns([1,1])
    with col_a:
        # call kept the same signature to avoid impacting other code
        render_profile("Andre Oneal A. Plaza", "profile_andre", "upload_andre")
    with col_b:
        render_profile("Rica Baliling", "profile_rica", "upload_rica")

    st.markdown("---")
    st.markdown("all god to be glory")
    st.write("Developed for a capstone project.")

# End of file
