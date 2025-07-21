from setuptools import setup, find_packages

# A list of dependencies needed for the project, matching requirements.txt
install_requires = [
    # Core ML & Transformers
    "accelerate==0.32.0",
    "bitsandbytes==0.41.1",
    "huggingface-hub==0.23.3",
    "numpy==1.24.3",
    "optimum==1.23.0",
    "pandas==2.2.2",
    "sentence-transformers==2.7.0",
    "auto-gptq==0.7.1",
    "transformers==4.41.2",

    # LangChain Ecosystem
    "langchain==0.2.3",
    "langchain-core==0.2.3",
    "langchain-community==0.2.3",
    "langchain-huggingface==0.0.3",

    # Data & Web
    "yfinance==0.2.45",
    "requests==2.32.3",
    "pypdf==4.3.0",

    # Forex & Currency
    "forex-python==1.8",
    "pycountry==23.12.11",
    "CurrencyConverter==0.17.14",
    "Babel==2.15.0",

    # Other Utilities
    "pinecone-client==3.2.2",
    "python-dotenv==1.0.0",
    "cryptography==41.0.2",
]

setup(
    name="finance_rag",
    version="0.1.0",
    description="A Finance RAG project using LangChain, Pinecone, and transformers.",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=install_requires,
    entry_points={
        "console_scripts": [
            "store_index=store_index:main"
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
)