# Rusty Machine

**Rusty Machine** is a high-performance machine learning library built with Rust and CUDA, exposed through a clean and simple Python API. It is designed to leverage the massive parallelism of modern NVIDIA GPUs to dramatically accelerate the training of classical machine learning models.

This project was developed as a semester project to explore the intersection of low-level systems programming, GPU architecture, and machine learning algorithms.

---

## 🚀 Features

* **GPU-Accelerated:** Core computations are executed on the GPU using custom CUDA kernels and optimized libraries like cuBLAS and cuSOLVER.
* **High Performance:** Achieves a significant speedup (often >10-20x) over traditional CPU-bound libraries like Scikit-learn for large datasets.
* **Clean Python API:** Provides a simple, Scikit-learn-like interface for training and prediction, making it easy to integrate into existing Python workflows.
* **Robust Implementations:** Includes production-ready solvers for:
    * **Linear Regression:** Using the Normal Equation method with GPU-accelerated matrix inversion.
    * **Logistic Regression:** Using a hyper-optimized Mini-Batch Gradient Descent solver.

---

## ベンチマーク Performance Benchmark

The primary goal of Rusty Machine is to deliver a significant performance improvement over CPU-based libraries. The following benchmark was run on a dataset of 500,000 samples and 100 features.

| Metric              | Rusty Machine (GPU) | Scikit-learn (CPU) |
| ------------------- | ------------------- | ------------------ |
| **Accuracy** | ~80%                | ~80%               |
| **Training Time** | **~0.45 seconds** | ~9.0 seconds       |
| **Performance Gain**| **~20x** | 1.0x               |

*Results are illustrative and depend on hardware and dataset size.*

---

## 🛠️ Installation

1.  **Prerequisites:**
    * Python 3.8+
    * Rust toolchain (including `cargo`)
    * NVIDIA CUDA Toolkit (version 12.x recommended)

2.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd rusty-machine
    ```

3.  **Create and activate a Python virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

4.  **Install dependencies and the library:**
    ```bash
    pip install . scikit-learn
    maturin develop
    ```

---

## Usage

The API is designed to be familiar to users of Scikit-learn.

```python
from rustymachine_api.models import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# 1. Generate data
X, y = make_classification(n_samples=100000, n_features=50)
X_train, X_test, y_train, y_test = train_test_split(X, y)

# 2. Initialize and train the model
model = LogisticRegression(epochs=100, lr=0.01, batch_size=512)
model.fit(X_train, y_train)

# 3. Make predictions
predictions = model.predict(X_test)

print(f"Model Accuracy: {accuracy_score(y_test, predictions):.4f}")