import streamlit as st
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification, make_regression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn.linear_model import LinearRegression as SklearnLinearRegression
from sklearn.metrics import accuracy_score, r2_score
from rustymachine_api.models import LogisticRegression as RustyLogisticRegression
from rustymachine_api.models import LinearRegression as RustyLinearRegression

# --- Page Configuration ---
st.set_page_config(
    page_title="Rusty Machine // Final Presentation",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Styling ---
st.markdown("""
<style>
    .main .block-container { padding-top: 2rem; padding-bottom: 3rem; }
    .stMetric {
        border: 1px solid #444;
        border-radius: 10px;
        padding: 1rem;
        background-color: #1E1E1E;
    }
    .result-header {
        font-size: 1.75rem;
        font-weight: bold;
        color: #00A0B0;
        text-align: center;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# --- Training Functions ---
@st.cache_data
def run_benchmarks(model_type, n_samples, n_features):
    # Generate data
    if model_type == "Logistic Regression":
        X, y = make_classification(n_samples=n_samples, n_features=n_features, n_informative=int(n_features*0.8), flip_y=0.05, random_state=42)
        metric_name = "Accuracy"
    else:
        X, y = make_regression(n_samples=n_samples, n_features=n_features, n_informative=int(n_features*0.8), noise=25, random_state=42)
        metric_name = "R² Score"
    
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- Train Rusty Machine ---
    if model_type == "Logistic Regression":
        model_rm = RustyLogisticRegression(epochs=100, lr=0.05, batch_size=512, random_state=42)
    else:
        model_rm = RustyLinearRegression()
    
    start_rm = time.time()
    model_rm.fit(X_train_scaled, y_train)
    duration_rm = time.time() - start_rm
    preds_rm = model_rm.predict(X_test_scaled)
    score_rm = accuracy_score(y_test, preds_rm) if model_type == "Logistic Regression" else r2_score(y_test, preds_rm)

    # --- Train Scikit-learn ---
    if model_type == "Logistic Regression":
        model_sk = SklearnLogisticRegression(solver='saga', max_iter=100, tol=1e-3, random_state=42)
    else:
        model_sk = SklearnLinearRegression()

    start_sk = time.time()
    model_sk.fit(X_train_scaled, y_train)
    duration_sk = time.time() - start_sk
    preds_sk = model_sk.predict(X_test_scaled)
    score_sk = accuracy_score(y_test, preds_sk) if model_type == "Logistic Regression" else r2_score(y_test, preds_sk)
    
    results = {
        "metric_name": metric_name,
        "rm_score": score_rm, "sk_score": score_sk,
        "rm_duration": duration_rm, "sk_duration": duration_sk,
    }
    return results

# --- UI Layout ---
st.title("🚀 Rusty Machine: A High-Performance ML Library")
st.markdown("A demonstration of GPU-accelerated machine learning with Rust and CUDA.")
st.markdown("---")

# --- OFFICIAL BENCHMARK RESULTS (STATIC) ---
st.header("🏆 Official Benchmark Results")
st.markdown("The following results were achieved by running the full-scale benchmark on a large dataset, demonstrating the significant performance gains of the `rusty-machine` library.")

st.subheader("Linear Regression (1,000,000 Samples)")
col1, col2, col3 = st.columns(3)
col1.metric("Rusty Machine Time", "0.978s", delta="-73.5%", delta_color="inverse")
col2.metric("Scikit-learn Time", "3.927s")
col3.metric("Performance Gain", "4.01x")
st.metric("Model R² Score", "0.9979 (Identical for both models)")


st.subheader("Logistic Regression (500,000 Samples)")
col1, col2, col3 = st.columns(3)
col1.metric("Rusty Machine Time", "0.517s", delta="-92.1%", delta_color="inverse")
col2.metric("Scikit-learn Time", "6.515s")
col3.metric("Performance Gain", "12.61x")
st.metric("Model Accuracy", "0.794 (Identical for both models)")
st.markdown("---")


# --- LIVE DEMO SECTION ---
st.header("⚙️ Live Demonstration")
st.info("Run a smaller, live benchmark to see the system in action. Due to resource constraints in live demos, dataset sizes are limited.")

with st.form("live_demo"):
    model_type_live = st.selectbox("Select Model Type", ("Logistic Regression", "Linear Regression"))
    
    if model_type_live == "Logistic Regression":
        # Use smaller, safer values for the live demo
        n_samples_live = st.slider("Samples (Live Demo)", 5_000, 50_000, 20_000, 5_000)
        n_features_live = st.slider("Features (Live Demo)", 10, 50, 20, 5)
    else:
        n_samples_live = st.slider("Samples (Live Demo)", 10_000, 100_000, 50_000, 10_000)
        n_features_live = st.slider("Features (Live Demo)", 10, 50, 20, 5)

    submitted = st.form_submit_button("Run Live Benchmark", use_container_width=True, type="primary")

if submitted:
    st.markdown("<p class='result-header'>Live Benchmark Results</p>", unsafe_allow_html=True)
    with st.spinner(f"Running live benchmark for {model_type_live}..."):
        results = run_benchmarks(model_type_live, n_samples_live, n_features_live)
    
    st.subheader("Performance Comparison")
    col1, col2, col3 = st.columns(3)
    col1.metric("Rusty Machine Time", f"{results['rm_duration']:.3f}s")
    col2.metric("Scikit-learn Time", f"{results['sk_duration']:.3f}s")
    col3.metric("Performance Gain", f"{results['sk_duration']/results['rm_duration']:.2f}x")
    
    st.subheader("Accuracy Comparison")
    col1, col2 = st.columns(2)
    col1.metric(f"Rusty Machine {results['metric_name']}", f"{results['rm_score']:.4f}")
    col2.metric(f"Scikit-learn {results['metric_name']}", f"{results['sk_score']:.4f}")