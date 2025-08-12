from setuptools import setup, find_packages

setup(
    name="sheria-kiganjani",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "anthropic>=0.5.0",
        "fastapi>=0.104.1",
        "pytest>=7.4.3",
        "python-dotenv>=1.0.0",
        "uvicorn>=0.24.0",
        "pydantic>=2.5.2",
        "pytest-asyncio>=0.21.1",
        "httpx>=0.25.2",
    ],
    python_requires=">=3.8",
)
