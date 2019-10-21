from setuptools import setup, find_packages

setup(
    name="evolutionary",
    packages=find_packages(),
    install_requires=["torch", "numpy", "matplotlib", "pre-commit", "pandas", "pyyaml"],
)
