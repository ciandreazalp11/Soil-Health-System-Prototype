import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score, confusion_matrix
from io import BytesIO
import joblib
import time

st.set_page_config(
    page_title="🌱 Soil Health ML App",
    layout="wide",
    page_icon="🌿"
)

# ----------------- THEME SETTINGS (Dynamic) -----------------
# Define themes
theme_classification = {
    "background_main": "linear-gradient(120deg, #0f2c2c 0%, #1a4141 40%, #0e2a2a 100%)",
    "sidebar_bg": "rgba(15, 30, 30, 0.9)",
    "primary_color": "#81c784", # Green for accents
    "secondary_color": "#a5d6a7", # Lighter green for text
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

theme_regression = {
    "background_main": "linear-gradient(120deg, #3d2d38 0%, #5e4a55 40%, #3a2b37 100%)",
    "sidebar_bg": "rgba(40, 30, 38, 0.9)",
    "primary_color": "#f06292", # Pink for accents
    "secondary_color": "#f8bbd0", # Lighter pink for text
    "button_gradient": "linear-gradient(90deg, #ec407a, #d81b60)",
    "button_text": "#2e1c27",
    "header_glow_color_1": "#f06292",
    "header_glow_color_2": "#d81b60",
    "menu_icon_color": "#f06292",
    "nav_link_color": "#ffe0e7",
    "nav_link_selected_bg": "#d81b60",
    "info_bg": "#422137",
    "info_border": "#f06292",
    "success_bg": "#5c2e4f",
    "success_border": "#f06292",
    "warning_bg": "#5c502e",
    "warning_border": "#dcd380",
    "error_bg": "#5c2e2e",
    "error_border": "#ef9a9a",
    "text_color": "#ffe0e7",
    "title_color": "#f8bbd0",
}

# Session state for current theme
if "current_theme" not in st.session_state:
    st.session_state["current_theme"] = theme_classification # Default

# Function to apply theme
def apply_theme(theme):
    st.markdown(f"""
    <link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;600;700&family=Playfair+Display:wght@400;700&display=swap" rel="stylesheet">
    <style>
    .stApp {{ font-family: 'Montserrat', sans-serif; color:{theme["text_color"]}; min-height:100vh; background: {theme["background_main"]}; background-attachment: fixed; }}
    section[data-testid="stSidebar"] {{ background: {theme["sidebar_bg"]} !important; border-radius:12px; padding:18px; box-shadow: 0 12px 40px rgba(0,0,0,0.6); }}
    div[data-testid="stSidebarNav"] a {{ color:{theme["nav_link_color"]} !important; border-radius:8px; padding:10px; }}
    div[data-testid="stSidebarNav"] a:hover {{ background: rgba(255,255,255,0.06) !important; transform: translateX(6px); transition: all .18s; }}
    h1,h2,h3,h4,h5,h6 {{ color:{theme["title_color"]}; text-shadow: 0 0 10px {theme["primary_color"]}20; font-family: 'Playfair Display', serif; }}
    .stButton button {{ background: {theme["button_gradient"]} !important; color:{theme["button_text"]} !important; border-radius:12px; padding:0.55rem 1rem; font-weight:700; box-shadow: 0 8px 20px {theme["primary_color"]}30; transition: all .2s; }}
    .stButton button:hover {{ transform: scale(1.04); box-shadow: 0 12px 30px {theme["primary_color"]}40; }}
    .legend {{ background: rgba(0,0,0,0.4); border-radius:12px; padding:10px; color:{theme["text_color"]}; font-size:14px; margin-top:15px; border: 1px solid {theme["primary_color"]}50; }}
    .legend span {{ display: inline-block; width: 14px; height: 14px; border-radius: 50%; margin-right: 6px; vertical-align: middle; }}
    .header-glow {{ position:absolute; right:20px; top:12px; width:78px; height:78px; opacity:0.95; animation: shimmer 8s linear infinite; }}
    @keyframes shimmer {{ 0% {{ transform: rotate(0deg) translateY(0) }} 50% {{ transform: rotate(6deg) translateY(-10px) }} 100% {{ transform: rotate(0deg) translateY(0) }} }}
    .footer {{ text-align:center; color:{theme["secondary_color"]}; font-size:13px; padding:10px; margin-top:18px; }}
    
    /* Info/Success/Warning/Error boxes */
    div[data-testid="stInfo"] {{ background-color: {theme["info_bg"]}; border-left: 5px solid {theme["info_border"]}; color: {theme["text_color"]}; }}
    div[data-testid="stSuccess"] {{ background-color: {theme["success_bg"]}; border-left: 5px solid {theme["success_border"]}; color: {theme["text_color"]}; }}
    div[data-testid="stWarning"] {{ background-color: {theme["warning_bg"]}; border-left: 5px solid {theme["warning_border"]}; color: {theme["text_color"]}; }}
    div[data-testid="stError"] {{ background-color: {theme["error_bg"]}; border-left: 5px solid {theme["error_border"]}; color: {theme["text_color"]}; }}

    /* Selectbox */
    div[data-testid="stSelectbox"] div[data-testid="stMarkdownContainer"] p {{ color:{theme["secondary_color"]}; }}
    div[data-testid="stSelectbox"] div[data-testid="stInputContainer"] {{ background-color: rgba(255,255,255,0.08); border-radius:8px; border:1px solid {theme["primary_color"]}30; }}
    div[data-testid="stSelectbox"] div[data-testid="stInputContainer"] div {{ color:{theme["text_color"]}; }}

    /* Slider */
    .stSlider > div > div > div:nth-child(1) {{ background: {theme["primary_color"]}40; }} /* Track background */
    .stSlider > div > div > div:nth-child(2) {{ background: {theme["primary_color"]}; }} /* Progress bar */
    .stSlider > div > div > div:nth-child(3) {{ background: {theme["primary_color"]}; border: 1px solid {theme["secondary_color"]}; }} /* Thumb */
    .stSlider label p {{ color:{theme["secondary_color"]}; }}

    /* Radio button */
    div[data-testid="stRadio"] label p {{ color:{theme["secondary_color"]}; }}
    div[data-testid="stRadio"] div[role="radiogroup"] label span {{ background-color: rgba(255,255,255,0.1); border: 1px solid {theme["primary_color"]}; }}
    div[data-testid="stRadio"] div[role="radiogroup"] label[data-testid="stRadio"] div[data-testid="stMarkdownContainer"] p {{ color:{theme["text_color"]}; }}
    div[data-testid="stRadio"] div[role="radiogroup"] label[data-testid="stRadio"] div[data-testid="stMarkdownContainer"] {{ background-color: rgba(255,255,255,0.1); border: 1px solid {theme["primary_color"]}; border-radius: 8px; padding: 5px 10px; margin: 5px 0; }}
    div[data-testid="stRadio"] div[role="radiogroup"] label[data-testid="stRadio"] input:checked + div {{ background-color: {theme["primary_color"]} !important; border: 1px solid {theme["secondary_color"]}; }}
    </style>

    <div style="position:relative;">
      <svg class="header-glow" viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg">
        <circle cx="32" cy="32" r="28" fill="{theme["header_glow_color_1"]}" opacity="0.08"/>
        <path fill="{theme["header_glow_color_2"]}" d="M20 44c6-12 24-24 40-28-4 18-18 34-34 34-1 0-3-1-6-6z" opacity="0.28"/>
      </svg>
    </div>
    """, unsafe_allow_html=True)

# Apply the initial theme
apply_theme(st.session_state["current_theme"])

# ----------------- SIDEBAR MENU -----------------
with st.sidebar:
    selected = option_menu(
        "🌱 Soil Health App",
        ["📂 Upload Data", "📊 Visualization", "🤖 Modeling", "📈 Results", "🌿 Insights"],
        icons=["cloud-upload", "bar-chart", "robot", "graph-up", "lightbulb"],
        menu_icon="list",
        default_index=0,
        styles={"container": {"padding": "5!important", "background-color": st.session_state["current_theme"]["sidebar_bg"]},
                "icon": {"color": st.session_state["current_theme"]["menu_icon_color"], "font-size": "20px"},
                "nav-link": {"color": st.session_state["current_theme"]["nav_link_color"], "font-size": "16px"},
                "nav-link-selected": {"background-color": st.session_state["current_theme"]["nav_link_selected_bg"]},}
    )

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

if "df" not in st.session_state:
    st.session_state["df"] = None
if "results" not in st.session_state:
    st.session_state["results"] = None
if "model" not in st.session_state:
    st.session_state["model"] = None
if "y_train_quantiles" not in st.session_state:
    st.session_state["y_train_quantiles"] = None
if "task_mode" not in st.session_state:
    st.session_state["task_mode"] = "Classification" # Default mode

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
        # Use qcut for more balanced bins if possible
        fert = pd.qcut(df[col], q=q, labels=labels, duplicates='drop')
        if fert.nunique() < 3: # Fallback if qcut creates fewer than 3 unique bins
            fert = pd.cut(df[col], bins=3, labels=labels)
    except Exception:
        # Fallback to cut if qcut fails entirely (e.g., all values are same)
        fert = pd.cut(df[col], bins=3, labels=labels, include_lowest=True)
    return fert.astype(str)

def interpret_label(label):
    l = str(label).lower()
    if l in ["high", "good", "healthy", "3", "2.0"]: # Ensure 2.0 is covered if float values
        return ("Good", "green", "✅ Nutrients are balanced. Ideal for most crops.")
    if l in ["moderate", "medium", "2", "1.0"]:
        return ("Moderate", "orange", "⚠️ Some nutrient imbalance. Consider minor adjustments.")
    return ("Poor", "red", "🚫 Deficient or problematic — take corrective action.")

# ----------------- UPLOAD DATA -----------------
if selected == "📂 Upload Data":
    st.title("📂 Upload Soil Data")
    st.markdown("Upload your soil analysis datasets here (.csv or .xlsx). The app will automatically clean and preprocess the data for analysis.")
    
    uploaded_files = st.file_uploader("Select multiple datasets", type=['csv', 'xlsx'], accept_multiple_files=True, key="uploader")
    
    if st.session_state["df"] is not None and not uploaded_files:
        st.info(f"✅ A preprocessed dataset is already loaded ({st.session_state['df'].shape[0]} rows, {st.session_state['df'].shape[1]} cols).")
        st.dataframe(st.session_state["df"].head())
        if st.button("🔁 Clear current dataset and upload new ones", help="This will remove the current dataset and allow you to upload new files."):
            st.session_state["df"] = None
            st.session_state["results"] = None # Clear results if data changes
            st.session_state["model"] = None # Clear model if data changes
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
                st.success(f"✅ Cleaned: **{file.name}** ({df_file.shape[0]} rows, kept cols: {', '.join(cols_to_keep)})")
            except Exception as e:
                st.warning(f"⚠️ Skipped **{file.name}**: {e}")
        
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
                    # Fallback for categories if mode fails (e.g., all unique)
                    df[c].fillna(method='ffill', inplace=True)
            
            df.dropna(how='all', inplace=True)
            st.session_state["df"] = df
            
            st.subheader("🔗 Final Merged, Cleaned & Preprocessed Dataset Preview")
            st.write(f"**Rows:** {df.shape[0]} — **Columns:** {df.shape[1]}")
            st.dataframe(df.head())
            download_df_button(df, filename="final_preprocessed_soil_dataset.csv", label="⬇️ Download Cleaned & Preprocessed Data")
            st.success("✨ Auto preprocessing applied and dataset saved for further analysis.")
            st.info("💡 You can now proceed to 'Visualization' or 'Modeling' tabs.")
        else:
            st.error("No valid datasets could be processed. Please check file formats and column names.")

# ----------------- VISUALIZATION -----------------
elif selected == "📊 Visualization":
    st.title("📊 Soil Data Visualization")
    st.markdown("Explore the distributions and relationships within your soil data.")

    if "df" in st.session_state and st.session_state["df"] is not None:
        df = st.session_state["df"]
        
        # Ensure 'Nitrogen' is available for fertility label creation in plots
        if 'Nitrogen' in df.columns and 'Fertility_Level' not in df.columns:
            df['Fertility_Level'] = create_fertility_label(df, col='Nitrogen', q=3)

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            st.warning("No numeric columns available for plotting. Please check your data.")
        else:
            st.subheader("Distribution of Soil Parameters")
            col1, col2 = st.columns(2)
            with col1:
                feature_dist = st.selectbox("Select a numeric feature for Distribution", numeric_cols, key="dist_feature")
            
            if feature_dist:
                fig_hist = px.histogram(df, x=feature_dist, nbins=30, marginal="box",
                                        color_discrete_sequence=[st.session_state["current_theme"]["primary_color"]],
                                        title=f"Distribution of {feature_dist}")
                fig_hist.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_hist, use_container_width=True)
            
            st.subheader("🌐 Feature Correlation Heatmap")
            corr = df[numeric_cols].corr() # Only numeric columns for correlation
            fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale=px.colors.sequential.Viridis,
                                 title="Correlation Matrix of Soil Parameters")
            fig_corr.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                   coloraxis_colorbar=dict(title="Correlation"))
            st.plotly_chart(fig_corr, use_container_width=True)

            st.subheader("Scatter Plot: Feature Relationships")
            col_x = st.selectbox("X-axis Feature", numeric_cols, index=0, key="scatter_x")
            col_y = st.selectbox("Y-axis Feature", numeric_cols, index=1 if len(numeric_cols) > 1 else 0, key="scatter_y")
            
            if col_x and col_y:
                fig_scatter = px.scatter(df, x=col_x, y=col_y,
                                         color='Fertility_Level' if 'Fertility_Level' in df.columns else None,
                                         color_discrete_map={'Low': 'red', 'Moderate': 'orange', 'High': 'green'},
                                         title=f"{col_x} vs. {col_y}",
                                         hover_data=df.columns)
                fig_scatter.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_scatter, use_container_width=True)

            st.subheader("Box Plots for Outlier Detection")
            box_feature = st.selectbox("Select a numeric feature for Box Plot", numeric_cols, key="box_feature")
            if box_feature:
                fig_box = px.box(df, y=box_feature,
                                 color_discrete_sequence=[st.session_state["current_theme"]["primary_color"]],
                                 title=f"Box Plot of {box_feature}")
                fig_box.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_box, use_container_width=True)

    else:
        st.info("Please upload and preprocess data first in the 'Upload Data' tab to enable visualizations.")

# ----------------- MODELING (Random Forest only) -----------------
elif selected == "🤖 Modeling":
    st.title("🤖 Modeling & Prediction Using Random Forest")
    st.markdown("Configure and train your Random Forest model for soil analysis.")

    if "df" not in st.session_state or st.session_state["df"] is None:
        st.info("Please upload data first in the 'Upload Data' tab.")
    else:
        df = st.session_state["df"].copy() # Use a copy to avoid modifying original session state DF directly

        # --- Mode Selection ---
        st.subheader("🎯 Select Prediction Task")
        current_task_mode = st.radio(
            "Choose your model's objective:",
            ["Classification", "Regression"],
            index=0 if st.session_state["task_mode"] == "Classification" else 1,
            key="task_selector_radio",
            help="Classification predicts soil fertility categories (Low, Moderate, High). Regression predicts the exact Nitrogen level."
        )

        # Update theme based on selected task
        if current_task_mode != st.session_state["task_mode"]:
            st.session_state["task_mode"] = current_task_mode
            st.session_state["current_theme"] = theme_classification if current_task_mode == "Classification" else theme_regression
            st.experimental_rerun() # Rerun to apply new theme immediately

        st.markdown(f"<p style='color:{st.session_state['current_theme']['secondary_color']}; font-size:1.1em;'>Current Mode: <strong>{st.session_state['task_mode']}</strong></p>", unsafe_allow_html=True)
        
        # Ensure 'Nitrogen' is available
        if 'Nitrogen' not in df.columns:
            st.error("❗ The 'Nitrogen' column is required for modeling (it's used as the target variable). Please ensure your dataset contains 'Nitrogen'.")
            st.stop()

        # Prepare target variable 'y'
        if st.session_state["task_mode"] == "Classification":
            st.info("Model will predict **Soil Fertility Level** (Low, Moderate, High) based on Nitrogen levels.")
            df['Fertility_Level'] = create_fertility_label(df, col='Nitrogen', q=3)
            y = df['Fertility_Level']
            X = df.drop(columns=['Nitrogen', 'Fertility_Level'], errors='ignore')
        else: # Regression
            st.info("Model will predict the **exact Nitrogen level** in the soil.")
            y = df['Nitrogen']
            X = df.drop(columns=['Nitrogen', 'Fertility_Level'], errors='ignore') # Ensure Fertility_Level is dropped if it was created
        
        # Select features
        available_features = X.select_dtypes(include=[np.number]).columns.tolist()
        if not available_features:
            st.error("No numeric features available after dropping the target. Please check your dataset.")
            st.stop()
        
        selected_features = st.multiselect(
            "Select features (input variables) for the model:",
            options=available_features,
            default=available_features,
            help="Choose the soil parameters you want the model to use for prediction."
        )

        if not selected_features:
            st.warning("Please select at least one feature to train the model.")
            st.stop()

        X = X[selected_features]

        # --- Hyperparameters ---
        st.subheader("⚙️ Random Forest Hyperparameters")
        col_hp1, col_hp2 = st.columns(2)
        with col_hp1:
            n_estimators = st.slider("Number of Trees (n_estimators)", 50, 500, 100, step=50,
                                     help="The number of decision trees in the forest.")
        with col_hp2:
            max_depth = st.slider("Max Depth (max_depth)", 2, 50, 10, step=1,
                                  help="The maximum depth of each tree. Prevents overfitting.")

        # Scaling and splitting
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        X = pd.DataFrame(X_scaled, columns=X.columns)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if st.button(f"🚀 Train {st.session_state['task_mode']} Model", help="Click to start training the Random Forest model."):
            with st.spinner(f"🧠 Training Random Forest ({st.session_state['task_mode']} mode)..."):
                time.sleep(0.5) # Simulate training time
                
                if st.session_state["task_mode"] == "Classification":
                    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1)
                else: # Regression
                    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1)
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                st.session_state["results"] = {
                    "task": st.session_state["task_mode"],
                    "y_test": y_test.tolist(),
                    "y_pred": y_pred.tolist(),
                    "model_name": "Random Forest",
                    "X_columns": X.columns.tolist(),
                    "feature_importances": model.feature_importances_.tolist(),
                    "feature_names": X.columns.tolist()
                }
                st.session_state["model"] = model
                
                if st.session_state["task_mode"] == "Regression":
                    st.session_state["y_train_quantiles"] = df['Nitrogen'].quantile([0.33, 0.66]).tolist()
                
                st.success("✅ Random Forest training completed! Proceed to 📈 Results to view performance.")
                # st.balloons() # Visual confirmation
        
        st.subheader("Make New Predictions (Optional)")
        st.markdown("Enter values for new soil samples to get predictions from your trained model.")
        if st.session_state["model"]:
            # Display input fields for each feature the model was trained on
            new_data_input = {}
            for feature in st.session_state["results"]["X_columns"]:
                new_data_input[feature] = st.number_input(f"Enter value for {feature}", value=0.0, format="%.2f", key=f"new_data_{feature}")

            if st.button("🔮 Predict New Sample", help="Get a prediction for the entered soil parameters."):
                # Convert input to DataFrame, scale, and predict
                input_df = pd.DataFrame([new_data_input])
                
                # IMPORTANT: Scale new data using the SAME scaler fitted on training data
                # We need to re-fit a scaler if we want to reuse it here.
                # For simplicity here, we'll refit a new scaler only on the selected features from the original df for consistent scaling range
                # A more robust solution would save and load the original scaler.
                
                temp_scaler = MinMaxScaler()
                temp_scaler.fit(df[st.session_state["results"]["X_columns"]]) # Fit on original range of selected features
                input_scaled = temp_scaler.transform(input_df)
                input_scaled_df = pd.DataFrame(input_scaled, columns=st.session_state["results"]["X_columns"])

                new_prediction = st.session_state["model"].predict(input_scaled_df)
                
                st.subheader("Prediction for New Sample:")
                if st.session_state["task_mode"] == "Classification":
                    pred_label, pred_color, pred_explanation = interpret_label(new_prediction[0])
                    st.markdown(f"**Predicted Soil Fertility Level:** <span style='color:{pred_color}; font-size:1.2em;'>**{pred_label}**</span>", unsafe_allow_html=True)
                    st.write(pred_explanation)
                else:
                    st.markdown(f"**Predicted Nitrogen Level:** <span style='color:{st.session_state['current_theme']['primary_color']}; font-size:1.2em;'>**{new_prediction[0]:.2f}**</span>", unsafe_allow_html=True)
        else:
            st.info("Train a model first to enable new predictions.")

# ----------------- RESULTS -----------------
elif selected == "📈 Results":
    st.title("📈 Model Results & Soil Health Interpretation")
    st.markdown("View the performance of your trained Random Forest model and get insights into soil health.")

    if not st.session_state.get("results"):
        st.info("Please train a model first in the 'Modeling' tab to see results.")
    else:
        results = st.session_state["results"]
        task = results["task"]
        y_test = np.array(results["y_test"])
        y_pred = np.array(results["y_pred"])
        
        if len(y_test) != len(y_pred):
            st.error("⚠️ Mismatch between test and prediction lengths.
