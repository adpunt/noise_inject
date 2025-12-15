"""
Setup script for NoiseInject package
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="noiseInject",
    version="0.1.0",
    author="Adelaide",
    description="Framework for testing ML model robustness to label noise",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19.0",
        "pandas>=1.1.0",
        "scipy>=1.5.0",
        "scikit-learn>=0.24.0",
    ],
    extras_require={
        "examples": [
            "rdkit>=2020.09.1",
            "torch>=1.9.0",
            "torch-geometric>=2.0.0",
            "xgboost>=1.5.0",
            "deepchem>=2.6.0",
        ],
        "all": [
            "rdkit>=2020.09.1",
            "torch>=1.9.0",
            "torch-geometric>=2.0.0",
            "scikit-learn>=0.24.0",
            "xgboost>=1.5.0",
            "deepchem>=2.6.0",
        ],
    },
)