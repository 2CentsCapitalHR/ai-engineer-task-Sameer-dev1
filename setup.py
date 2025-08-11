#!/usr/bin/env python3
"""
Setup script for ADGM Corporate Agent
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="adgm-corporate-agent",
    version="2.0.0",
    author="ADGM Corporate Agent Team",
    author_email="support@adgm-corporate-agent.com",
    description="AI-powered document analysis and compliance checking system for ADGM corporate documents",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/adgm-corporate-agent",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Legal Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Office/Business",
        "Topic :: Text Processing",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "adgm-agent=app:main",
            "adgm-ingest=ingest_adgm_sources:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.json", "*.txt", "*.md"],
    },
    keywords="adgm, corporate, legal, compliance, document-analysis, ai, rag",
    project_urls={
        "Bug Reports": "https://github.com/your-org/adgm-corporate-agent/issues",
        "Source": "https://github.com/your-org/adgm-corporate-agent",
        "Documentation": "https://adgm-corporate-agent.readthedocs.io/",
    },
)
