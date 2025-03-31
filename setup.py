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
    "numpy>=1.20.0",
    "scipy>= 1.3.1",
    "pandas>= 0.25.3",
    "tqdm>=4.41.1",
    "joblib>= 0.14.0",
    "matplotlib>= 3.2.2",
    "seaborn>= 0.11.0",
    "rapidfuzz>= 0.13.1",
    "scikit_learn>= 0.22.2.post1"
]

gensim_packages = [
    "gensim>=4.0.0"
]

sbert_packages = [
    "sentence-transformers>=0.4.1"
]

fast_cosine = [
    "sparse_dot_topn<1.0; python_version < '3.8'",
    "sparse_dot_topn>=1.1.5; python_version >= '3.8'",
]

embeddings_packages = [
    "torch>=1.4.0", 
    "flair>= 0.7"
]

spacy_packages = [
    "spacy>=3.0.1"
]

use_packages = [
    "tensorflow",
    "tensorflow_hub",
    "tensorflow_text"
]


extra_packages = embeddings_packages + fast_cosine + sbert_packages + spacy_packages + use_packages

dev_packages = docs_packages + test_packages + extra_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="polyfuzz",
    packages=find_packages(exclude=["notebooks", "docs"]),
    version="0.4.2",
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
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    install_requires=base_packages,
    extras_require={
        "test": test_packages,
        "docs": docs_packages,
        "dev": dev_packages,
        "flair": embeddings_packages,
        "fast": fast_cosine,
        "sbert": sbert_packages,
        "use": use_packages,
        "gensim": gensim_packages,
    },
    python_requires='>=3.9',
)
