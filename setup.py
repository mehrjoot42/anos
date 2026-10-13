from setuptools import setup, find_packages

setup(
    name="anos",
    version="0.2.0",
    description="Adaptive Proxy Selection for Censored Networks (improved)",
    author="ANOS contributors",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=["numpy>=1.24", "scipy>=1.11"],
    extras_require={"rl": ["torch>=2.1"]},
)
