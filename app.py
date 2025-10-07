import streamlit as st
import numpy as np
import cupy as cp
import pandas as pd
import time
import psutil
import os
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification, make_regression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn.linear_model import LinearRegression as SklearnLinearRegression
from sklearn.metrics import accuracy_score, r2_score

from rustymachine_api.models import LogisticRegression as RustyLogisticRegression
from rustymachine_api.models import LinearRegression as RustyLinearRegression

st.set_page_config(
    page_title="Rusty Machine // Performance Benchmark",
    layout="wide"
)

st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .title {
        font-family: 'monospace', sans-serif;
        color: #FFFFFF;
        text-align: center;
        padding: 1rem;
    }
    .metric-card {
        background-color: #1E1E1E;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 0.5rem;
        border: 1px solid #444444;
        text-align: center;
    }
    .metric-card h3 {
        color: #00A0B0;
        margin-bottom: 0.5rem;
    }
    .metric-card p {
        font-size: 2.5rem;
        color: #FFFFFF;
        font-weight: bold;
        margin: 0;
    }
    h1, h2, h3 {
        color: #CCCCCC;
    }
</style>
""", unsafe_allow_html=True)

def format_bytes(byte_count):
    if byte_count is None or byte_count == 0: return "0.00 B"
    power = 1024
    n = 0
    power_labels = {0: 'B', 1: 'KB', 2: 'MB', 3: 'GB'}
    while byte_count >= power and n < len(power_labels) - 1:
        byte_count /= power
        n += 1
    return f"{byte_count:.2f} {power_labels[n]}"

def train_rusty_model(model_type, X_train, y_train, epochs, lr, batch_size):
    try:
        if model_type == "Logistic Regression":
            model = RustyLogisticRegression(epochs=epochs, lr=lr, batch_size=batch_size, random_state=42)
        else:
            model = RustyLinearRegression()
        
        # ✅ **THE FIX**: Pass NumPy arrays to the fit method as it now handles conversion.
        X_np = np.asarray(X_train)
        y_np = np.asarray(y_train)

        # Measure GPU memory usage by creating temporary CuPy arrays
        gpu_mem_used = cp.asarray(X_np).nbytes + cp.asarray(y_np).nbytes

        start_time = time.time()
        model.fit(X_np, y_np)
        duration = time.time() - start_time
        
        return duration, model, gpu_mem_used
    except Exception as e:
        st.error(f"Error in Rusty Machine: {e}")
        return -1, None, 0

def train_sklearn_model(model_type, X_train, y_train, epochs):
    try:
        if model_type == "Logistic Regression":
            model = SklearnLogisticRegression(max_iter=epochs, solver='saga', tol=1e-3, random_state=42)
        else:
            model = SklearnLinearRegression()
        
        cpu_mem_used = X_train.nbytes + y_train.nbytes

        start_time = time.time()
        model.fit(X_train, y_train.ravel())
        duration = time.time() - start_time
        
        return duration, model, cpu_mem_used
    except Exception as e:
        st.error(f"Error in Scikit-learn: {e}")
        return -1, None, 0

st.markdown('<h1 class="title">Rusty Machine // Performance Benchmark</h1>', unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #AAAAAA;'>An academic showcase of GPU-accelerated machine learning with Rust and CUDA.</h3>", unsafe_allow_html=True)
st.markdown("---")

with st.sidebar:
    st.header("Benchmark Configuration")
    
    model_type = st.selectbox(
        "Select Model Type",
        ("Logistic Regression", "Linear Regression")
    )
    
    if model_type == "Logistic Regression":
        default_samples, default_features = 500000, 100
    else:
        default_samples, default_features = 1000000, 100
    
    n_samples = st.slider(
        "Dataset Samples",
        min_value=10000, max_value=1000000, value=default_samples, step=10000
    )
    
    n_features = st.slider(
        "Dataset Features",
        min_value=10, max_value=200, value=default_features, step=10
    )

    if model_type == "Logistic Regression":
        st.subheader("Hyperparameters")
        epochs = st.slider("Epochs", 50, 500, 200, 10)
        learning_rate = st.slider("Learning Rate", 0.001, 0.1, 0.05, 0.001, format="%.3f")
        batch_size = st.select_slider("Batch Size", [128, 256, 512, 1024, 2048], 1024)
    else:
        epochs, learning_rate, batch_size = 0, 0, 0
    
    run_button = st.button("Initiate Benchmark", use_container_width=True, type="primary")

if not run_button:
    st.info("Configure the benchmark in the sidebar and click 'Initiate Benchmark'.")

if run_button:
    with st.spinner(f"Generating data for {model_type}..."):
        if model_type == "Logistic Regression":
            X, y = make_classification(n_samples=n_samples, n_features=n_features, n_informative=int(n_features*0.8), random_state=42)
        else:
            X, y = make_regression(n_samples=n_samples, n_features=n_features, n_informative=int(n_features*0.8), noise=25, random_state=42)

        X = X.astype(np.float32)
        y = y.astype(np.float32)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

    st.success(f"Dataset generated with {n_samples:,} samples and {n_features} features.")
    st.markdown("---")

    st.header("Benchmark in Progress...")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Rusty Machine (GPU)")
        with st.spinner("Training..."):
            rusty_duration, rusty_model, gpu_mem = train_rusty_model(model_type, X_train_scaled, y_train, epochs, learning_rate, batch_size)
        st.success(f"Completed in {rusty_duration:.4f}s")
    
    with col2:
        st.subheader("Scikit-learn (CPU)")
        with st.spinner("Training..."):
            sklearn_duration, sklearn_model, cpu_mem = train_sklearn_model(model_type, X_train_scaled, y_train, epochs)
        st.success(f"Completed in {sklearn_duration:.4f}s")

    st.markdown("---")
    
    st.header("Final Results")
    
    if rusty_model and sklearn_model:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f'<div class="metric-card"><h3>Rusty Machine Time</h3><p>{rusty_duration:.3f}s</p></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-card"><h3>Scikit-learn Time</h3><p>{sklearn_duration:.3f}s</p></div>', unsafe_allow_html=True)
        with col3:
            speedup = sklearn_duration / rusty_duration if rusty_duration > 0 else 0
            st.markdown(f'<div class="metric-card"><h3>Performance Gain</h3><p>{speedup:.2f}x</p></div>', unsafe_allow_html=True)
        
        st.markdown("### Model Performance Comparison")
        if model_type == "Logistic Regression":
            rusty_preds = rusty_model.predict(X_test_scaled)
            sklearn_preds = sklearn_model.predict(X_test_scaled)
            rusty_score = accuracy_score(y_test, rusty_preds)
            sklearn_score = accuracy_score(y_test, sklearn_preds)
            metric_name = "Accuracy"
        else:
            rusty_preds = rusty_model.predict(X_test_scaled)
            sklearn_preds = sklearn_model.predict(X_test_scaled)
            rusty_score = r2_score(y_test, rusty_preds)
            sklearn_score = r2_score(y_test, sklearn_preds)
            metric_name = "R² Score"

        score_data = {
            'Model': ['Rusty Machine (GPU)', 'Scikit-learn (CPU)'],
            metric_name: [f"{rusty_score:.4f}", f"{sklearn_score:.4f}"]
        }
        score_df = pd.DataFrame(score_data)
        st.table(score_df)

        st.markdown("### Resource Usage Comparison")
        mem_data = {
            'Model': ['Rusty Machine (GPU)', 'Scikit-learn (CPU)'],
            'Memory (Data)': [format_bytes(gpu_mem), format_bytes(cpu_mem)]
        }
        mem_df = pd.DataFrame(mem_data)
        st.table(mem_df)

        st.markdown("### Prediction Comparison (First 5 Samples)")
        pred_data = {
            'Feature 1': X_test_scaled[:5, 0],
            'Feature 2': X_test_scaled[:5, 1],
            'Feature 3': X_test_scaled[:5, 2],
            'Rusty Machine': rusty_preds.flatten()[:5],
            'Scikit-learn': sklearn_preds.flatten()[:5],
        }
        pred_df = pd.DataFrame(pred_data).round(2)
        st.table(pred_df)