"""
Rusty Machine: A High-Performance Machine Learning Library in Rust.

This package provides GPU-accelerated implementations of common machine learning
algorithms, accessible through a clean Python API.
"""

from .models import LinearRegression, LogisticRegression

__all__ = [
    "LinearRegression",
    "LogisticRegression"
]