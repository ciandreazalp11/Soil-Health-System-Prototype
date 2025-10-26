# (Full file begins)
# app.py
import streamlit as st
import pandas as pd
import numpy as np
@@ -17,6 +17,7 @@
import io as sysio
import os

# page config
st.set_page_config(
    page_title="Machine Learning-Driven Soil Analysis for Sustainable Agriculture System",
    layout="wide",
@@ -92,9 +93,12 @@
    st.session_state["profile_andre"] = None
if "profile_rica" not in st.session_state:
    st.session_state["profile_rica"] = None
# navigation index for sidebar
if "page_index" not in st.session_state:
    st.session_state["page_index"] = 0

# UI navigation override used by "Proceed" buttons on Home
if "page_override" not in st.session_state:
    st.session_state["page_override"] = None
if "last_sidebar_selected" not in st.session_state:
    st.session_state["last_sidebar_selected"] = None

# ----------------- THEME APPLIER + BACKGROUND -----------------
def apply_theme(theme):
@@ -132,7 +136,6 @@ def apply_theme(theme):
      box-shadow: 0 12px 40px rgba(0,0,0,0.12);
      z-index: 2;
    }}
    /* modernized header card in sidebar */
    .sidebar-header {{
      padding: 12px;
      border-radius: 10px;
@@ -142,7 +145,6 @@ def apply_theme(theme):
    }}
    .sidebar-title {{ font-family: 'Playfair Display', serif; color:{theme['title_color']}; margin:0; }}
    .sidebar-sub {{ font-size:12px; color:{theme['secondary_color']}; margin-top:6px; opacity:0.95; }}
    /* make the menu links a bit more spaced and larger hit area */
    div[data-testid="stSidebarNav"] a {{
      color:{theme['nav_link_color']} !important;
      border-radius:8px;
@@ -180,27 +182,14 @@ def apply_theme(theme):
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

# ----------------- SIDEBAR (modeling first) -----------------
with st.sidebar:
    st.markdown(
        f"""
@@ -213,30 +202,31 @@ def apply_theme(theme):
    )
    st.write("---")

    default_index = st.session_state.get("page_index", 0)
    # note: Modeling is first now
    selected = option_menu(
        None,
        PAGES,
        ["🏠 Home", "🤖 Modeling", "📊 Visualization", "📈 Results", "🌿 Insights", "👤 About"],
        icons=["house", "robot", "bar-chart", "graph-up", "lightbulb", "person-circle"],
        menu_icon="list",
        default_index=default_index,
        default_index=0,
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

    # update last selection to clear overrides if user uses sidebar
    if st.session_state["last_sidebar_selected"] != selected:
        st.session_state["page_override"] = None
        st.session_state["last_sidebar_selected"] = selected

# determine page (sidebar selection or override from Home "Proceed" buttons)
page = st.session_state["page_override"] if st.session_state["page_override"] else selected

# ----------------- COMMON SETTINGS -----------------
column_mapping = {
    'pH': ['pH', 'ph', 'Soil_pH'],
@@ -289,14 +279,10 @@ def pil_to_base64(img: Image.Image, fmt="PNG"):

def render_profile(name, session_key, upload_key):
    """
    REPLACED function: now uses static images from assets/ folder (andre.png / rica.png)
    and displays a neon-glow circular avatar. Uploaders and hints removed.
    Signature kept the same to avoid touching other code that calls this function.
    Kept signature same. Displays avatar from assets folder or placeholder.
    uploaders removed per earlier instruction.
    """
    # container
    st.markdown("<div style='display:flex;flex-direction:column;align-items:center;text-align:center;'>", unsafe_allow_html=True)

    # determine filename by name
    image_filename = None
    if "Andre" in name:
        image_filename = "andre.png"
@@ -313,7 +299,6 @@ def render_profile(name, session_key, upload_key):
            except Exception:
                img_b64 = None

    # neon glow CSS (keeps visual style but isolated here)
    st.markdown("""
    <style>
    .neon-glow {
@@ -352,14 +337,11 @@ def render_profile(name, session_key, upload_key):
        """

    st.markdown(html, unsafe_allow_html=True)

    # name + centered BSIS 4-A (replaces uploader hint + "Upload an image..." text)
    st.markdown(f"<div style='margin-top:8px;font-weight:700;color:{st.session_state['current_theme']['secondary_color']};'>{name}</div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:14px;color:rgba(255,255,255,0.85);margin-top:4px;font-weight:600;'>BSIS 4-A</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ----------------- Reusable upload & preprocess function (used on Home) -----------------
# ----------------- Upload & Preprocess widget -----------------
def upload_and_preprocess_widget():
    st.markdown("### 📂 Upload Soil Data")
    st.markdown("Upload one or more soil analysis files (.csv or .xlsx). The app will attempt to standardize column names and auto-preprocess numeric columns.")
@@ -420,64 +402,93 @@ def upload_and_preprocess_widget():
            st.write(f"Rows: {df.shape[0]} — Columns: {df.shape[1]}")
            st.dataframe(df.head())
            download_df_button(df)

            # show proceed buttons
            st.markdown("---")
            st.markdown("When you're ready you can go straight to Modeling or Visualization:")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("➡️ Proceed to Modeling"):
                    st.session_state["page_override"] = "🤖 Modeling"
                    st.experimental_rerun()
            with col2:
                if st.button("➡️ Proceed to Visualization"):
                    st.session_state["page_override"] = "📊 Visualization"
                    st.experimental_rerun()
        else:
            st.error("No valid sheets processed. Check file formats and column headers.")

# ----------------- HOME -----------------
if selected == "🏠 Home":
if page == "🏠 Home":
    st.title("Machine Learning-Driven Soil Analysis for Sustainable Agriculture System")
    st.markdown("<small style='color:rgba(255,255,255,0.75)'>Capstone Project</small>", unsafe_allow_html=True)
    st.write("---")

    # No task-mode toggle on Home (moved to Modeling). Home only handles upload + navigation.
    # Upload UI only (no mode toggle here)
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
elif page == "🤖 Modeling":
    st.title("🤖 Modeling — Random Forest")
    st.markdown("Fine tune hyperparameters and train Random Forest models for Classification or Regression.")

    st.markdown("Train Random Forest models for Fertility (Regression) or Soil Health (Classification).")
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
        # ---- Mode toggle: checkbox (stateful) + visual switch that changes color ----
        st.markdown("#### Model Mode")
        # model_mode_checkbox True => Regression, False => Classification
        default_checkbox = True if st.session_state.get("task_mode") == "Regression" else False
        chk = st.checkbox("Switch to Regression mode", value=default_checkbox, key="model_mode_checkbox")
        if chk:
            st.session_state["task_mode"] = "Regression"
            st.session_state["current_theme"] = theme_sakura
        else:
            st.session_state["task_mode"] = "Classification"
            st.session_state["current_theme"] = theme_classification
        apply_theme(st.session_state["current_theme"])

        # Visual switch (reflects checkbox state) — purely visual
        switch_color = "#ff8aa2" if st.session_state["task_mode"] == "Regression" else "#81c784"
        st.markdown(f"""
        <style>
        .fake-switch {{
            width:70px;
            height:36px;
            border-radius:20px;
            background:{switch_color};
            display:inline-block;
            position:relative;
            box-shadow: 0 6px 18px rgba(0,0,0,0.25);
        }}
        .fake-knob {{
            width:28px;height:28px;border-radius:50%;
            background:rgba(255,255,255,0.95); position:absolute; top:4px;
            transition: all .18s ease;
        }}
        .knob-left {{ left:4px; }}
        .knob-right {{ right:4px; }}
        .switch-label {{ font-weight:600; margin-left:10px; color:{st.session_state['current_theme']['text_color']}; }}
        </style>
        <div style="display:flex;align-items:center;margin-bottom:10px;">
          <div class="fake-switch">
            <div class="fake-knob {'knob-right' if st.session_state['task_mode']=='Regression' else 'knob-left'}"></div>
          </div>
          <div class="switch-label">{'Regression' if st.session_state['task_mode']=='Regression' else 'Classification'}</div>
        </div>
        """, unsafe_allow_html=True)

        if 'Nitrogen' not in df.columns:
            st.error("Missing 'Nitrogen' column required as target. Ensure your dataset contains 'Nitrogen'.")
            st.stop()
        st.markdown("---")

        # prepare target and features
        # prepare target and X depending on mode
        if st.session_state["task_mode"] == "Classification":
            df['Fertility_Level'] = create_fertility_label(df, col='Nitrogen', q=3)
            y = df['Fertility_Level']
            if 'Nitrogen' in df.columns:
                df['Fertility_Level'] = create_fertility_label(df, col='Nitrogen', q=3)
            y = df['Fertility_Level'] if 'Fertility_Level' in df.columns else None
        else:
            y = df['Nitrogen']
            y = df['Nitrogen'] if 'Nitrogen' in df.columns else None

        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        # drop Nitrogen from features
        if 'Nitrogen' in numeric_features:
            numeric_features.remove('Nitrogen')

@@ -487,299 +498,276 @@ def upload_and_preprocess_widget():

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
            X = df[selected_features]

            st.subheader("Hyperparameters")
            col1, col2 = st.columns(2)
            with col1:
                n_estimators = st.slider("n_estimators", 50, 500, 150, step=50)
            with col2:
                max_depth = st.slider("max_depth", 2, 50, 12)

            # scaling & split
            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(X)
            X_scaled_df = pd.DataFrame(X_scaled, columns=selected_features)
            test_size = st.slider("Test set fraction (%)", 10, 40, 20, step=5)
            X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y, test_size=test_size/100, random_state=42)

            if st.button("🚀 Train Model"):
                with st.spinner("Training Random Forest..."):
                    time.sleep(0.25)
                    if st.session_state["task_mode"] == "Classification":
                        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1)
                    else:
                        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1)

                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    # cross-validation summary
                    try:
                        cv_scores = cross_val_score(model, X_scaled_df, y, cv=5, scoring='accuracy' if st.session_state["task_mode"] == "Classification'".strip("'") else 'r2')
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
                        "cv_summary": cv_summary
                    }
                    st.session_state["trained_on_features"] = selected_features
                    st.success("✅ Training completed. Go to 'Results' to inspect performance.")

            # Predict inline
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

# ----------------- VISUALIZATION -----------------
elif selected == "📊 Visualization":
elif page == "📊 Visualization":
    st.title("📊 Data Visualization")
    st.markdown("Explore distributions, correlations, and relationships in your preprocessed data.")
    if st.session_state["df"] is None:
        st.info("Please upload data first in 'Home' (Upload Data is integrated there).")
    else:
        df = st.session_state["df"].copy()
        # Ensure fertility label for classification visualizations exists if needed
        df = st.session_state["df"]
        # ensure fertility label for classification view
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
        # Parameter overview (histograms + level boxes)
        st.subheader("Parameter Overview (Levels & Distributions)")
        param_cols = [c for c in ['pH', 'Nitrogen', 'Phosphorus', 'Potassium', 'Moisture', 'Organic Matter'] if c in df.columns]
        if not param_cols:
            st.warning("No recognized parameter columns found. Required example columns: pH, Nitrogen, Phosphorus, Potassium, Moisture, Organic Matter")
        else:
            st.info("Not enough numeric columns to generate a correlation heatmap.")

        st.write("---")
        st.markdown("### Mode-specific Visualizations")
            # grid of hist + box for each
            for col in param_cols:
                fig = px.histogram(df, x=col, nbins=30, marginal="box", title=f"Distribution: {col}", color_discrete_sequence=[st.session_state["current_theme"]["primary_color"]])
                fig.update_layout(template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)
                # small explanation
                st.markdown(f"<div style='font-size:13px;color:rgba(255,255,255,0.85)'>This histogram shows the distribution of **{col}** across samples. Use the median and spread to assess central tendency and variability.</div>", unsafe_allow_html=True)
                st.markdown("---")

            # Correlation heatmap
            st.subheader("Correlation Matrix")
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                corr = df[numeric_cols].corr()
                fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale=px.colors.sequential.Viridis, title="Correlation Heatmap")
                fig_corr.update_layout(template="plotly_dark")
                st.plotly_chart(fig_corr, use_container_width=True)
                st.markdown("<div style='font-size:13px;color:rgba(255,255,255,0.85)'>Correlation between numeric parameters. High correlation may indicate redundant features for modeling.</div>", unsafe_allow_html=True)

        # Mode-specific visualization
        st.markdown("---")
        st.subheader("Mode-specific Visuals")
        mode = st.session_state.get("task_mode", "Classification")
        st.markdown(f"Current Mode: **{mode}**. (Change mode in Modeling.)")

        st.markdown(f"**Current Mode:** {mode}")
        if mode == "Classification":
            # Show fertility-level colored histograms / stacked distributions
            # Show Fertility_Level counts + stacked bar of distribution by parameter ranges
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
                df['Fertility_Level'] = create_fertility_label(df, col='Nitrogen', q=3) if 'Nitrogen' in df.columns else "Unknown"
            fig_bar = px.histogram(df, x='Fertility_Level', title="Fertility Level Counts")
            fig_bar.update_layout(template="plotly_dark")
            st.plotly_chart(fig_bar, use_container_width=True)
            st.markdown("<div style='font-size:13px;color:rgba(255,255,255,0.85)'>Shows counts of Low, Moderate, High fertility across samples.</div>", unsafe_allow_html=True)
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
            # Regression visuals — show relationship Nitrogen vs top features
            if 'Nitrogen' not in df.columns:
                st.warning("Nitrogen column required for Regression visuals.")
            else:
                st.info("No regression model results available. Train a Regression model in Modeling first.")
                # show scatter of Nitrogen vs top numeric columns
                top_num = [c for c in df.select_dtypes(include=[np.number]).columns if c != 'Nitrogen'][:3]
                for feat in top_num:
                    fig_sc = px.scatter(df, x=feat, y='Nitrogen', trendline="ols", title=f"Nitrogen vs {feat}")
                    # try to safely compute trendline (statsmodels required). Plotly will warn if missing.
                    try:
                        fig_sc.update_layout(template="plotly_dark")
                    except Exception:
                        pass
                    st.plotly_chart(fig_sc, use_container_width=True)
                    st.markdown(f"<div style='font-size:13px;color:rgba(255,255,255,0.85)'>Scatter showing relationship of {feat} against Nitrogen. Trendline (OLS) included if statsmodels is available.</div>", unsafe_allow_html=True)

# ----------------- RESULTS -----------------
elif selected == "📈 Results":
elif page == "📈 Results":
    st.title("📈 Model Results & Interpretation")
    if not st.session_state.get("results"):
        st.info("No trained model in session. Train a model first (Modeling).")
        st.info("No trained model in session. Train a model first (Modeling or Quick Model).")
    else:
        results = st.session_state["results"]
        task = results["task"]
        y_test = np.array(results["y_test"])
        y_pred = np.array(results["y_pred"])

        # Model summary card
        st.markdown("<div class='card'><strong>Model Summary</strong></div>", unsafe_allow_html=True)
        col_a, col_b = st.columns([2,1])
        with col_a:
        st.subheader("Model Summary")
        colA, colB = st.columns([3,2])
        with colA:
            st.write(f"**Model:** {results.get('model_name','Random Forest')}")
            st.write(f"**Features:** {', '.join(results.get('X_columns',[]))}")
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
        with colB:
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

        st.markdown("---")

        # Two-column metrics + explanations (Option 1 you chose)
        metrics_col, explain_col = st.columns([2,1])
        with metrics_col:
            st.subheader("Performance Metrics")
            if task == "Classification":
                # Accuracy metric
                try:
                    acc = accuracy_score(y_test, y_pred)
                    st.metric("Accuracy", f"{acc:.3f}")
                except Exception:
                    st.write("Accuracy N/A")
                # confusion matrix
                st.markdown("**Confusion Matrix**")
                try:
                    cm = confusion_matrix(y_test, y_pred, labels=['Low','Moderate','High'])
                    fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale=px.colors.sequential.Viridis, title="Confusion Matrix (Low / Moderate / High)")
                    fig_cm.update_layout(template="plotly_dark", height=350)
                    st.plotly_chart(fig_cm, use_container_width=True)
                except Exception:
                    st.write("Confusion matrix not available")
                # classification report as table
                st.markdown("#### 📊 Classification Report (Detailed)")
                try:
                    rep = classification_report(y_test, y_pred, output_dict=True)
                    rep_df = pd.DataFrame(rep).transpose().reset_index()
                    rep_df.rename(columns={"index":"Class"}, inplace=True)
                    cols_order = ["Class","precision","recall","f1-score","support"]
                    # ensure columns present
                    rep_df = rep_df[[c for c in cols_order if c in rep_df.columns]]
                    # style and display
                    styled = rep_df.style.format({
                        "precision":"{:.2f}",
                        "recall":"{:.2f}",
                        "f1-score":"{:.2f}",
                        "support":"{:.0f}"
                    }).background_gradient(subset=["f1-score"] if "f1-score" in rep_df.columns else None, cmap="Greens")
                    st.dataframe(styled, use_container_width=True)
                    st.markdown("<div style='font-size:13px;color:rgba(255,255,255,0.85)'>Precision/Recall/F1 per class. Support = number of samples.</div>", unsafe_allow_html=True)
                except Exception as e:
                    st.text(classification_report(y_test,y_pred))

            else:
                # regression metrics
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                st.metric("RMSE", f"{rmse:.3f}")
                st.metric("MAE", f"{mae:.3f}")
                st.metric("R²", f"{r2:.3f}")

                # actual vs predicted
                df_res = pd.DataFrame({"Actual_Nitrogen": y_test, "Predicted_Nitrogen": y_pred})
                st.markdown("**Sample predictions**")
                st.dataframe(df_res.head(10), use_container_width=True)
                # scatter actual vs pred with trendline if possible
                st.markdown("**Actual vs Predicted**")
                try:
                    fig1 = px.scatter(df_res, x="Actual_Nitrogen", y="Predicted_Nitrogen", trendline="ols",
                                      title="Actual vs Predicted Nitrogen (Model Predictions)")
                    fig1.update_layout(template="plotly_dark")
                    st.plotly_chart(fig1, use_container_width=True)
                except Exception:
                    # fallback without trendline
                    fig1 = px.scatter(df_res, x="Actual_Nitrogen", y="Predicted_Nitrogen",
                                      title="Actual vs Predicted Nitrogen (no trendline available)")
                    fig1.update_layout(template="plotly_dark")
                    st.plotly_chart(fig1, use_container_width=True)

                # residual distribution
                df_res["residual"] = df_res["Actual_Nitrogen"] - df_res["Predicted_Nitrogen"]
                fig_res = px.histogram(df_res, x="residual", nbins=30, title="Residual Distribution")
                fig_res.update_layout(template="plotly_dark")
                st.plotly_chart(fig_res, use_container_width=True)

        with explain_col:
            st.subheader("What the metrics mean")
            if task == "Classification":
                st.markdown("- **Accuracy:** Overall fraction of correct predictions.")
                st.markdown("- **Confusion Matrix:** Rows = true classes, Columns = predicted classes.")
                st.markdown("- **Precision:** Of all predicted positive, how many were actually positive.")
                st.markdown("- **Recall:** Of all actual positive samples, how many were found.")
                st.markdown("- **F1-score:** Harmonic mean of precision and recall; balanced measure.")
            else:
                st.markdown("- **RMSE:** Root Mean Squared Error — lower is better; same units as target.")
                st.markdown("- **MAE:** Mean Absolute Error — average magnitude of errors.")
                st.markdown("- **R²:** Proportion of variance explained by the model (1 is perfect).")
            st.markdown("---")
            st.markdown("**Feature importances** (Top 5)")
            fi = results.get("feature_importances", [])
            feat = results.get("X_columns", [])
            if fi and feat:
                df_fi = pd.DataFrame({"feature": feat, "importance": fi}).sort_values("importance", ascending=False).head(5)
                st.table(df_fi.reset_index(drop=True))
            else:
                st.info("No feature importances available.")

# ----------------- INSIGHTS -----------------
elif selected == "🌿 Insights":
elif page == "🌿 Insights":
    st.title("🌿 Soil Health Insights")
    st.markdown("Automated soil health recommendations based on model outputs and feature signals.")
    if st.session_state["results"] is None:
@@ -807,19 +795,16 @@ def upload_and_preprocess_widget():
            st.info("No model-based insights available yet. Train a model first.")

# ----------------- ABOUT / PROFILE -----------------
elif selected == "👤 About":
elif page == "👤 About":
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
