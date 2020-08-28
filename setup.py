from setuptools import setup, find_packages

setup(
    name="evolutionary",
    packages=find_packages(),
    install_requires=[
        "torch>=1.6.0",
        "numpy",
        "matplotlib==3.1.3",
        "pre-commit",
        "pandas",
        "pyyaml",
    ],
)
