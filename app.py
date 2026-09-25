import pandas as pd
import numpy as np
import scipy.stats as stats
import streamlit as st
import json
import difflib

st.set_page_config(page_title="Data 'Splorer", page_icon="🌈", layout="wide")

# --- CUSTOM RAINBOW TITLE (COMIC SANS) ---
st.markdown("""
    <style>
    .rainbow-title {
        font-family: 'Comic Sans MS', 'Comic Sans', cursive;
        background-image: linear-gradient(to right, red, orange, yellow, green, blue, indigo, violet);
        -webkit-background-clip: text;
        color: transparent;
        font-size: 4em;
        font-weight: bold;
        text-align: center;
        margin-bottom: 0.2em;
    }
    </style>
    <div class="rainbow-title">Data 'Splorer</div>
    <hr>
""", unsafe_allow_html=True)

# --- STATE MANAGEMENT ---
if "config" not in st.session_state:
    st.session_state.config = {}
if "saved_groups" not in st.session_state:
    st.session_state.saved_groups = {}
if "num_filters" not in st.session_state:
    st.session_state.num_filters = 2

def add_filter(): st.session_state.num_filters += 1
def remove_filter(): 
    if st.session_state.num_filters > 1: st.session_state.num_filters -= 1

st.sidebar.header("📂 1. Settings & Data")
config_file = st.sidebar.file_uploader("Load Global Settings (.json)", type=["json"])
if config_file is not None:
    try:
        st.session_state.config = json.load(config_file)
        st.sidebar.success("Global settings loaded!")
    except Exception:
        st.sidebar.error("Error loading config.")

# --- 1. MEMORY-EFFICIENT PREVIEW UPLOAD ---
uploaded_file = st.sidebar.file_uploader("Upload your CSV data file", type=["csv"])

if uploaded_file is not None:
    @st.cache_data
    def load_columns(file):
        file.seek(0)
        return list(pd.read_csv(file, nrows=0).columns)
        
    @st.cache_data
    def load_preview(file, usecols):
        file.seek(0)
        return pd.read_csv(file, usecols=usecols, nrows=100, low_memory=True)

    all_columns = load_columns(uploaded_file)
    default_kept_cols = [c for c in st.session_state.config.get("kept_cols", all_columns) if c in all_columns]

    st.sidebar.markdown("### Global Variable List")
    st.sidebar.caption("Click 'X' to permanently exclude variables to save memory.")
    kept_columns = st.sidebar.multiselect(
        "Active Variables",
        all_columns,
        default=default_kept_cols,
    )

    if not kept_columns:
        st.stop()

    preview_df = load_preview(uploaded_file, kept_columns)
    columns = list(preview_df.columns)

    # --- 2. CONFIGURATION & STRUCTURE ---
    st.sidebar.header("🛠️ 2. Data Structure")
    case_default = st.session_state.config.get("case_id_col", columns[0] if columns else None)
    case_id_col = st.sidebar.selectbox(
        "Select Case ID Column",
        columns,
        index=columns.index(case_default) if case_default in columns else 0,
    )
    
    visit_default = st.session_state.config.get("visit_col", None)
    visit_col = st.sidebar.selectbox(
        "Select Visit / Timepoint Column",
        [None] + columns,
        index=([None] + columns).index(visit_default) if visit_default in [None] + columns else 0,
    )

    # --- 3. DATA CLEANING ---
    st.sidebar.header("🧹 3. Data Cleaning")
    custom_missing = st.sidebar.text_input(
        "Custom Missing Value Codes (e.g., -4, 999)",
        value=st.session_state.config.get("custom_missing", "")
    )

    enable_typo_fix = st.sidebar.checkbox(
        "Enable Batch Typo Correction", 
        value=st.session_state.config.get("enable_typo_fix", False)
    )

    selected_text_cols = []
    similarity_threshold = 85
    if enable_typo_fix:
        text_cols = preview_df.select_dtypes(include=["object"]).columns.tolist()
        typo_default = [c for c in st.session_state.config.get("typo_cols", []) if c in text_cols]
        selected_text_cols = st.sidebar.multiselect("Select Text Column(s)", text_cols, default=typo_default)
        similarity_threshold = st.sidebar.slider("Typo Threshold", 70, 100, st.session_state.config.get("typo_threshold", 85), 5)

    st.sidebar.markdown("---")
    current_config = {
        "kept_cols": kept_columns, "case_id_col": case_id_col, "visit_col": visit_col,
        "custom_missing": custom_missing, "enable_typo_fix": enable_typo_fix,
        "typo_cols": selected_text_cols, "typo_threshold": similarity_threshold
    }
    st.sidebar.download_button("💾 Download Setup (.json)", data=json.dumps(current_config, indent=4), file_name="setup.json", mime="application/json")

    # --- 4. MANUAL EXECUTION BUTTON ---
    st.sidebar.markdown("### Ready to Process?")
    if st.sidebar.button("🚀 Apply Settings & Process Data", type="primary"):
        with st.spinner("Loading full dataset into memory (This may take a moment)..."):
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, usecols=kept_columns, low_memory=True)
            for col in df.select_dtypes(include=["float64"]).columns:
                df[col] = pd.to_numeric(df[col], downcast="float")
            for col in df.select_dtypes(include=["int64"]).columns:
                df[col] = pd.to_numeric(df[col], downcast="integer")
            
            if custom_missing:
                missing_list = [x.strip() for x in custom_missing.split(",")]
                to_replace = []
                for x in missing_list:
                    try:
                        to_replace.append(float(x)) if '.' in x else to_replace.extend([int(x), float(x)])
                    except ValueError:
                        to_replace.append(x)
                df.replace(to_replace, np.nan, inplace=True)

            if enable_typo_fix and selected_text_cols:
                # Built-in difflib is used here for WebAssembly compatibility instead of rapidfuzz
                for col in selected_text_cols:
                    unique_vals = [str(x) for x in df[col].dropna().unique()]
                    canonical_map = {}
                    for val in unique_vals:
                        if not canonical_map:
                            canonical_map[val] = val
                            continue
                        matches = difflib.get_close_matches(val, list(canonical_map.values()), n=1, cutoff=similarity_threshold/100.0)
                        canonical_map[val] = matches[0] if matches else val
                    df[col] = df[col].map(canonical_map)

            st.session_state.processed_data = df
            st.success("Data successfully processed!")

    # --- 5. MAIN INTERFACE ---
    if "processed_data" in st.session_state:
        df_filtered = st.session_state.processed_data.copy()
        current_cols = list(df_filtered.columns)

        # --- BOOLEAN FILTER BUILDER ---
        st.markdown("### 🎛️ Boolean Filter Builder")
        
        col_fb1, col_fb2 = st.columns([1, 1])
        with col_fb1:
            filter_file = st.file_uploader("Load Boolean Filters (.json)", type=["json"])
            if filter_file:
                try:
                    loaded_filters = json.load(filter_file)
                    st.session_state.num_filters = len(loaded_filters)
                    for i, f in enumerate(loaded_filters):
                        st.session_state[f"filt_logic_{i}"] = f.get("logic", "AND")
                        st.session_state[f"filt_col_{i}"] = f.get("col")
                        st.session_state[f"filt_op_{i}"] = f.get("op")
                        st.session_state[f"filt_val_{i}"] = f.get("val")
                except Exception:
                    st.error("Invalid filter file.")

        current_filters = []
        for i in range(st.session_state.num_filters):
            cols = st.columns([1, 3, 2, 3])
            with cols[0]:
                logic = "START" if i == 0 else st.selectbox("Logic", ["AND", "OR"], key=f"filt_logic_{i}")
                if i == 0: st.markdown("**Start Condition**")
            with cols[1]:
                col = st.selectbox("Field", [""] + current_cols, key=f"filt_col_{i}")
            with cols[2]:
                op = st.selectbox("Operator", ["==", "!=", ">", "<", ">=", "<=", "Contains", "Is Missing", "Not Missing"], key=f"filt_op_{i}")
            with cols[3]:
                if op not in ["Is Missing", "Not Missing"]:
                    val = st.text_input("Value", key=f"filt_val_{i}")
                else:
                    val = ""
                    st.text_input("Value", value="N/A", disabled=True, key=f"filt_val_disabled_{i}")
            if col:
                current_filters.append({"logic": logic, "col": col, "op": op, "val": val})

        colA, colB, colC = st.columns(3)
        with colA: st.button("➕ Add Condition", on_click=add_filter)
        with colB: st.button("➖ Remove Condition", on_click=remove_filter)
        with colC: st.download_button("💾 Download Filter JSON", data=json.dumps(current_filters, indent=4), file_name="filters.json", mime="application/json")

        # Execute Boolean Filters
        if current_filters:
            mask = pd.Series(True, index=df_filtered.index)
            for i, f in enumerate(current_filters):
                c, op, v = f["col"], f["op"], f["val"]
                logic = f["logic"]
                
                if op == "Is Missing": temp_mask = df_filtered[c].isna()
                elif op == "Not Missing": temp_mask = df_filtered[c].notna()
                else:
                    if pd.api.types.is_numeric_dtype(df_filtered[c]):
                        try: v_cast = float(v)
                        except: v_cast = v
                    else: v_cast = v
                    
                    try:
                        if op == "==": temp_mask = df_filtered[c] == v_cast
                        elif op == "!=": temp_mask = df_filtered[c] != v_cast
                        elif op == ">": temp_mask = df_filtered[c] > v_cast
                        elif op == "<": temp_mask = df_filtered[c] < v_cast
                        elif op == ">=": temp_mask = df_filtered[c] >= v_cast
                        elif op == "<=": temp_mask = df_filtered[c] <= v_cast
                        elif op == "Contains": temp_mask = df_filtered[c].astype(str).str.contains(str(v), case=False, na=False)
                    except TypeError:
                        temp_mask = pd.Series(False, index=df_filtered.index)
                        
                if i == 0 or logic == "START": mask = temp_mask
                elif logic == "AND": mask = mask & temp_mask
                elif logic == "OR": mask = mask | temp_mask
            df_filtered = df_filtered[mask]

        long_apply = st.checkbox("Keep entire history (all visits) for matching Case IDs", value=False)
        if long_apply and case_id_col:
            matching_cases = df_filtered[case_id_col].dropna().unique()
            df_filtered = st.session_state.processed_data[st.session_state.processed_data[case_id_col].isin(matching_cases)]

        st.markdown("---")
        
        # Calculate N and n sizes
        n_obs = len(df_filtered)
        if case_id_col and case_id_col in df_filtered.columns:
            n_cases = df_filtered[case_id_col].nunique()
            n_string = f"Cases (N) = {n_cases:,} | Observations (n) = {n_obs:,}"
        else:
            n_string = f"Observations (n) = {n_obs:,}"
        
        st.markdown(f"### 📈 Current Sample Size: `{n_string}`")

        # --- SAVE GROUP COHORT ---
        with st.expander("💾 Save Current View as Cohort / Group"):
            g_name = st.text_input("Group Name (e.g., 'Aspirin Users', 'Age > 65')")
            if st.button("Save Group"):
                if g_name:
                    st.session_state.saved_groups[g_name] = df_filtered.copy()
                    st.success(f"Saved Group '{g_name}' with {n_string}")
                else:
                    st.warning("Please enter a group name.")

        tab1, tab2, tab3, tab4 = st.tabs(["Data Preview", "Descriptive Statistics", "Inferential Stats", "Longitudinal View"])

        with tab1:
            st.write(f"**Previewing Dataset:** {n_string}")
            st.dataframe(df_filtered.head(100), use_container_width=True)

        with tab2:
            st.subheader(f"Descriptive Statistics ({n_string})")
            numeric_cols = df_filtered.select_dtypes(include=["number"]).columns.tolist()
            cat_cols = df_filtered.select_dtypes(include=["object", "category"]).columns.tolist()
            analysis_type = st.radio("Select Variable Type", ["Numeric (Mean, SD, Range)", "Categorical (Frequencies)"])

            if analysis_type == "Numeric (Mean, SD, Range)" and numeric_cols:
                selected_num = st.multiselect("Select Numeric Variables", numeric_cols, default=numeric_cols[:2] if len(numeric_cols)>1 else numeric_cols)
                if selected_num:
                    desc = df_filtered[selected_num].describe().T
                    desc["range"] = desc["max"] - desc["min"]
                    st.dataframe(desc[["mean", "std", "min", "max", "range", "count"]], use_container_width=True)
                    
            elif analysis_type == "Categorical (Frequencies)" and cat_cols:
                selected_cat = st.selectbox("Select Categorical Variable", cat_cols)
                if selected_cat:
                    freq_df = df_filtered[selected_cat].value_counts(dropna=False).reset_index()
                    freq_df.columns = [selected_cat, "Frequency (n)"]
                    freq_df["Percentage"] = (freq_df["Frequency (n)"] / n_obs) * 100
                    st.dataframe(freq_df, use_container_width=True)

        with tab3:
            st.subheader("Inferential Statistics")
            stat_test = st.selectbox("Choose Analysis", ["Correlation Matrix", "Independent Samples t-test"])

            if stat_test == "Correlation Matrix" and len(numeric_cols) >= 2:
                corr_vars = st.multiselect("Select Variables for Correlation", numeric_cols, default=numeric_cols[:4] if len(numeric_cols)>3 else numeric_cols)
                if len(corr_vars) >= 2:
                    st.write(f"*(Correlation N based on pairwise complete observations from {n_string})*")
                    corr_matrix = df_filtered[corr_vars].corr(method=st.selectbox("Method", ["pearson", "spearman"]))
                    st.dataframe(corr_matrix, use_container_width=True)

            elif stat_test == "Independent Samples t-test":
                test_mode = st.radio("Comparison Mode", ["Split by Categorical Variable", "Compare Saved Groups"])
                
                if test_mode == "Split by Categorical Variable":
                    group_col = st.selectbox("Select Binary Grouping Variable", cat_cols + numeric_cols)
                    target_col = st.selectbox("Continuous Target Variable", numeric_cols)
                    if group_col and target_col:
                        unique_groups = df_filtered[group_col].dropna().unique()
                        if len(unique_groups) == 2:
                            g1 = df_filtered[df_filtered[group_col] == unique_groups[0]][target_col].dropna()
                            g2 = df_filtered[df_filtered[group_col] == unique_groups[1]][target_col].dropna()
                            st.write(f"**{unique_groups[0]}**: n = {len(g1):,} | **{unique_groups[1]}**: n = {len(g2):,}")
                            t_stat, p_val = stats.ttest_ind(g1, g2, nan_policy="omit")
                            st.metric("t-statistic", round(t_stat, 4))
                            st.metric("p-value", round(p_val, 4))
                        else:
                            st.warning("Selected grouping variable must have exactly 2 unique groups.")
                
                else: # Compare Saved Groups
                    if len(st.session_state.saved_groups) < 2:
                        st.warning("Please save at least 2 groups using the 'Save Current View as Cohort' tool above.")
                    else:
                        c1, c2, c3 = st.columns(3)
                        with c1: g1_name = st.selectbox("Group 1", list(st.session_state.saved_groups.keys()))
                        with c2: g2_name = st.selectbox("Group 2", list(st.session_state.saved_groups.keys()))
                        with c3: target_col = st.selectbox("Continuous Target Variable", numeric_cols)
                        
                        if g1_name and g2_name and g1_name != g2_name and target_col:
                            g1 = pd.to_numeric(st.session_state.saved_groups[g1_name][target_col], errors='coerce').dropna()
                            g2 = pd.to_numeric(st.session_state.saved_groups[g2_name][target_col], errors='coerce').dropna()
                            
                            st.write(f"**{g1_name}**: n = {len(g1):,} | **{g2_name}**: n = {len(g2):,}")
                            if len(g1) > 1 and len(g2) > 1:
                                t_stat, p_val = stats.ttest_ind(g1, g2, nan_policy="omit")
                                st.metric("t-statistic", round(t_stat, 4))
                                st.metric("p-value", round(p_val, 4))
                            else:
                                st.warning("Not enough valid observations to run a t-test.")

        with tab4:
            st.subheader(f"Longitudinal Case Trajectory ({n_string})")
            if case_id_col and visit_col:
                unique_cases = df_filtered[case_id_col].dropna().unique()
                if len(unique_cases) > 0:
                    selected_case = st.selectbox("Select Case ID to Track", unique_cases)
                    case_history = df_filtered[df_filtered[case_id_col] == selected_case].sort_values(by=visit_col)
                    st.dataframe(case_history, use_container_width=True)
            else:
                st.info("Please specify both Case ID and Visit columns to use the longitudinal view.")

else:
    st.info("👈 Please upload a CSV file in the sidebar to get started.")