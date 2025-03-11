from setuptools import setup, find_packages

setup(
    name="finance_rag",
    version="0.1.0",
    description="A Finance RAG project using LangChain, Pinecone, and sentence-transformers embeddings.",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "langchain==0.0.148",
        "pinecone-client==2.2.4",
        "sentence-transformers==2.2.2",
        "streamlit==1.22.0",
        "yfinance==0.2.18",
        "transformers==4.31.0",
        "requests==2.31.0"
    ],
    entry_points={
        "console_scripts": [
            # This creates a command line tool named "store_index" that runs the main function in store_index.py.
            "store_index=store_index:main"
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
