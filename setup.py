from setuptools import setup, find_packages


test_packages = [
    "pytest>=5.4.3",
    "pytest-cov>=2.6.1"
]

docs_packages = [
    "mkdocs==1.1",
    "mkdocs-material==4.6.3",
    "mkdocstrings==0.8.0",
]

base_packages = [
    "numpy>= 1.19.2",
    "scipy>= 1.3.1",
    "pandas>= 0.25.3",
    "tqdm>=4.48.2",
    "joblib>= 0.14.0",
    "sparse_dot_topn>=0.2.9",
    "matplotlib>= 3.3.3",
    "seaborn>= 0.11.0",
    "rapidfuzz>= 0.13.1",
    "scikit_learn>= 0.23.2"
]

embeddings_packages = ["flair>= 0.6.1.post1"]

extra_packages = embeddings_packages

dev_packages = docs_packages + test_packages + extra_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="polyfuzz",
    packages=find_packages(exclude=["notebooks", "docs"]),
    version="0.0.1",
    author="Maarten Grootendorst",
    author_email="maartengrootendorst@gmail.com",
    description="PolyFuzz performs fuzzy string matching, grouping, and evaluation.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    project_urls={
        "Documentation": "https://maartengr.github.io/polyfuzz/",
        "Source Code": "https://github.com/MaartenGr/PolyFuzz/",
        "Issue Tracker": "https://github.com/MaartenGr/PolyFuzz/issues",
    },
    url="https://github.com/MaartenGr/PolyFuzz",
    keywords="nlp string matching embeddings levenshtein tfidf bert",
    classifiers=[
        "Programming Language :: Python",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        "Operating System :: MacOS",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.8",
    ],
    install_requires=base_packages,
    extras_require={
        "test": test_packages,
        "docs": docs_packages,
        "dev": dev_packages,
        "flair": embeddings_packages,
        "all": extra_packages
    },
    python_requires='>=3.6',
)
