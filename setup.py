"""Enables installation as a Python package."""
from setuptools import setup, find_packages

setup(
    name='uwin_ai_assistant',
    version='0.1',
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "pandas",
        "openai",
        "qdrant-client",
        "fastapi",
    ]
)
