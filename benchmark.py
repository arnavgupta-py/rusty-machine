import time
import numpy as np
from sklearn.datasets import make_regression, make_classification
from sklearn.linear_model import LinearRegression as SklearnLinearRegression
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from rustymachine_api.models import LinearRegression as RustyLinearRegression
from rustymachine_api.models import LogisticRegression as RustyLogisticRegression


def print_header(title):
    print("=" * 80)
    print(f"=" * 80)


def print_results(model_name, score, duration, speedup=None):
    score_str = f"{score:.4f}"
    duration_str = f"{duration:.4f}s"
    speedup_str = f"{speedup:.2f}x" if speedup is not None else "1.00x"
    print(f"{model_name:<25} | {score_str:<25} | {duration_str:<25} | {speedup_str:<25}")


def run_linear_regression_benchmark():
    print_header("BENCHMARK 1: LINEAR REGRESSION")

    SAMPLES = 1_000_000
    FEATURES = 100
    INFORMATIVE_FEATURES = 75

    print(f"Generating data: {SAMPLES:,} samples, {FEATURES} features...\n")
    X, y = make_regression(
        n_samples=SAMPLES,
        n_features=FEATURES,
        n_informative=INFORMATIVE_FEATURES,
        noise=25,
        random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- Rusty Machine (GPU) ---
    print("Training Rusty Machine (GPU)...")
    model_rm = RustyLinearRegression()
    start_rm = time.time()
    model_rm.fit(X_train, y_train)
    duration_rm = time.time() - start_rm
    preds_rm = model_rm.predict(X_test)
    score_rm = r2_score(y_test, preds_rm)
    print(f"Training finished in {duration_rm:.4f} seconds.")

    # --- Scikit-learn (CPU) ---
    print("\nTraining Scikit-learn (CPU)...")
    model_sk = SklearnLinearRegression()
    start_sk = time.time()
    model_sk.fit(X_train, y_train)
    duration_sk = time.time() - start_sk
    preds_sk = model_sk.predict(X_test)
    score_sk = r2_score(y_test, preds_sk)
    print(f"Training finished in {duration_sk:.4f} seconds.")

    # --- Results ---
    print("\n" + "-" * 80)
    print(f"{'Model':<25} | {'R² Score':<25} | {'Training Time':<25} | {'Performance Gain':<25}")
    print("-" * 80)
    print_results("Rusty Machine (GPU)", score_rm, duration_rm, speedup=duration_sk / duration_rm)
    print_results("Scikit-learn (CPU)", score_sk, duration_sk)
    print("-" * 80)


def run_logistic_regression_benchmark():
    print_header("BENCHMARK 2: LOGISTIC REGRESSION")

    SAMPLES = 500_000
    FEATURES = 100
    INFORMATIVE_FEATURES = 50
    EPOCHS = 200
    LR = 0.05
    BATCH_SIZE = 1024

    print(f"Generating data: {SAMPLES:,} samples, {FEATURES} features...\n")
    X, y = make_classification(
        n_samples=SAMPLES,
        n_features=FEATURES,
        n_informative=INFORMATIVE_FEATURES,
        n_redundant=10,
        random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- Rusty Machine (GPU) ---
    print("Training Rusty Machine (GPU)...")
    model_rm = RustyLogisticRegression(
        epochs=EPOCHS, lr=LR, batch_size=BATCH_SIZE, random_state=42
    )
    start_rm = time.time()
    model_rm.fit(X_train_scaled, y_train)
    duration_rm = time.time() - start_rm
    preds_rm = model_rm.predict(X_test_scaled)
    score_rm = accuracy_score(y_test, preds_rm)
    print(f"Training finished in {duration_rm:.4f} seconds.")

    # --- Scikit-learn (CPU) ---
    print("\nTraining Scikit-learn (CPU)...")
    model_sk = SklearnLogisticRegression(
        solver='saga', max_iter=EPOCHS, tol=1e-4, random_state=42
    )
    start_sk = time.time()
    model_sk.fit(X_train_scaled, y_train)
    duration_sk = time.time() - start_sk
    preds_sk = model_sk.predict(X_test_scaled)
    score_sk = accuracy_score(y_test, preds_sk)
    print(f"Training finished in {duration_sk:.4f} seconds.")

    # --- Results ---
    print("\n" + "-" * 80)
    print(f"{'Model':<25} | {'Accuracy Score':<25} | {'Training Time':<25} | {'Performance Gain':<25}")
    print("-" * 80)
    print_results("Rusty Machine (GPU)", score_rm, duration_rm, speedup=duration_sk / duration_rm)
    print_results("Scikit-learn (CPU)", score_sk, duration_sk)
    print("-" * 80)


if __name__ == "__main__":
    run_linear_regression_benchmark()
    run_logistic_regression_benchmark()