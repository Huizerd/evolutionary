from setuptools import setup, find_packages

setup(
    name="evolutionary",
    packages=find_packages(),
    install_requires=[
        "torch",
        "deap",
        "numpy",
        "matplotlib",
        "pre-commit",
        "dask[bag]",
        "pandas",
        "pyyaml",
    ],
)
