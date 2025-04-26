from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="arxiv_search",
    version="0.1.0",
    author="You",
    author_email="your.email@example.com",
    description="ArXiv Paper Search and Analysis with LangGraph",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/arxiv-search-analysis",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "langgraph==0.3.34",
        "google-generativeai==0.8.5",
        "arxiv",
        "requests",
        "numpy",
        "beautifulsoup4",
        "chromadb",
        "ipython",
        "jupyter",
        "matplotlib",
        "pandas",
    ],
) 