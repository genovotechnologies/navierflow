from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="navierflow",
    version="1.0.0",
    author="NavierFlow Contributors",
    author_email="folabi@genovoteach.com",
    description="A high-performance fluid dynamics simulation engine with AI-powered optimizations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tafolabi009/navierflow",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    include_package_data=True,
    package_data={
        "navierflow": ["configs/*.yaml"],
    },
    entry_points={
        "console_scripts": [
            "navierflow=navierflow.last_shot:main",
            "navierflow-train=navierflow.ai.training.train:main",
        ],
    },
) 