import os
import io
import json
import tempfile
from typing import Tuple, List

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Phishing URL Detector", layout="wide")

MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/phishing_xgb_pipeline.joblib")
META_PATH  = os.getenv("META_PATH",  "artifacts/metadata.json")
IGNORED_INPUT_COLS = ['url', 'hostname', 'domain', 'status', 'Label', 'label']

@st.cache_resource(show_spinner=False)
def _load_artifacts_from_disk(model_path: str, meta_path: str):
    pipe = joblib.load(model_path)
    with open(meta_path, "r") as f:
        meta = json.load(f)
    numeric_cols = list(map(str, meta.get("numeric_cols", [])))
    categorical_cols = list(map(str, meta.get("categorical_cols", [])))
    label_source = str(meta.get("label_column_used", "status/label"))
    used_smote = bool(meta.get("used_smote", False))
    return pipe, numeric_cols, categorical_cols, label_source, used_smote, meta

def _persist_uploaded_artifact(uploaded_file, suffix: str) -> str:
    tmpdir = st.session_state.setdefault("_artifacts_dir", tempfile.mkdtemp(prefix="artifacts_"))
    path = os.path.join(tmpdir, f"uploaded_{suffix}")
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path

def _clean_and_align(df: pd.DataFrame, numeric_cols: List[str], categorical_cols: List[str]) -> Tuple[pd.DataFrame, List[str], List[str]]:
    dropped = [c for c in IGNORED_INPUT_COLS if c in df.columns]
    df = df.drop(columns=dropped, errors="ignore")
    df = df.replace([np.inf, -np.inf], np.nan)
    expected = list(numeric_cols) + list(categorical_cols)
    missing = [c for c in expected if c not in df.columns]
    for m in missing:
        if m in numeric_cols:
            df[m] = 0
        else:
            df[m] = ""
    df = df[expected].fillna(0)
    return df, dropped, missing

def _artifacts_loaded() -> bool:
    return ("pipe" in st.session_state and "numeric_cols" in st.session_state and "categorical_cols" in st.session_state)

st.sidebar.title("⚙️ Artifacts")
st.sidebar.caption("Load saved pipeline + metadata from repo or upload them here.")

if "attempted_disk_load" not in st.session_state:
    st.session_state.attempted_disk_load = True
    try:
        pipe, numeric_cols, categorical_cols, label_source, used_smote, meta = _load_artifacts_from_disk(MODEL_PATH, META_PATH)
        st.session_state.pipe = pipe
        st.session_state.numeric_cols = numeric_cols
        st.session_state.categorical_cols = categorical_cols
        st.session_state.label_source = label_source
        st.session_state.used_smote = used_smote
        st.session_state.meta = meta
        st.sidebar.success(f"Loaded artifacts from {MODEL_PATH} / {META_PATH}")
    except Exception as e:
        st.sidebar.warning(f"Could not load from disk: {e}")

st.sidebar.markdown("---")
up_m = st.sidebar.file_uploader("Upload pipeline (.joblib)", type=["joblib", "pkl"], key="upl_model")
up_j = st.sidebar.file_uploader("Upload metadata (.json)", type=["json"], key="upl_meta")

if st.sidebar.button("Use uploaded artifacts"):
    if up_m and up_j:
        try:
            mpath = _persist_uploaded_artifact(up_m, "pipeline.joblib")
            jpath = _persist_uploaded_artifact(up_j, "metadata.json")
            pipe, numeric_cols, categorical_cols, label_source, used_smote, meta = _load_artifacts_from_disk(mpath, jpath)
            st.session_state.pipe = pipe
            st.session_state.numeric_cols = numeric_cols
            st.session_state.categorical_cols = categorical_cols
            st.session_state.label_source = label_source
            st.session_state.used_smote = used_smote
            st.session_state.meta = meta
            st.sidebar.success("Uploaded artifacts loaded.")
        except Exception as e:
            st.sidebar.error(f"Failed to load uploaded artifacts: {e}")
    else:
        st.sidebar.error("Please upload both the pipeline and metadata files.")

st.title("🔍 Phishing URL Detector")
st.caption("XGBoost + One-Hot + Scaling + PCA (+SMOTE during training). Upload a CSV with the same feature columns used in training. If `url`/`status`/`label` are present, they'll be ignored.")

if _artifacts_loaded():
    with st.expander("Model information", expanded=False):
        st.write(f"**Used SMOTE (training):** `{st.session_state.used_smote}`")
        st.json(st.session_state.meta, expanded=False)

    threshold = st.slider("Decision threshold (probability for 'phishing')", 0.1, 0.9, 0.5, 0.01)
    show_probs = st.checkbox("Show probabilities", value=True)
    show_preview = st.checkbox("Show input preview (first 10 rows)", value=False)

    uploaded_csv = st.file_uploader("Upload CSV of samples", type=["csv"])
    if uploaded_csv is not None:
        try:
            raw = pd.read_csv(uploaded_csv)
        except Exception:
            uploaded_csv.seek(0)
            raw = pd.read_excel(uploaded_csv)

        if show_preview:
            st.subheader("Raw preview")
            st.dataframe(raw.head(10), use_container_width=True)

        X_aligned, dropped_cols, missing_cols = _clean_and_align(
            raw.copy(),
            st.session_state.numeric_cols,
            st.session_state.categorical_cols
        )

        st.info(f"Aligned to expected columns. Dropped: {dropped_cols or '[]'} | Added (filled) missing: {missing_cols or '[]'}")

        probs = st.session_state.pipe.predict_proba(X_aligned)[:, 1]
        preds = (probs >= threshold).astype(int)

        out = raw.copy()
        out["phishing_prob"] = probs
        out["prediction"] = np.where(preds == 1, "phishing", "legitimate")

        st.success(f"Predicted {(preds == 1).sum()} phishing out of {len(preds)} rows.")
        if show_probs:
            st.subheader("Predictions (head)")
            st.dataframe(out.head(20), use_container_width=True)

        st.subheader("Counts")
        st.write(out["prediction"].value_counts())

        csv_bytes = out.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download predictions", data=csv_bytes, file_name="predictions.csv", mime="text/csv")
    else:
        st.info("Upload a CSV to begin.")
else:
    st.error(
        "Artifacts not loaded. Either place the trained pipeline and metadata at "
        f"`{MODEL_PATH}` and `{META_PATH}` in your repo, or upload them from the sidebar."
    )
