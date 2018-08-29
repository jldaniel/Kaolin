from setuptools import setup, find_packages

setup(
    name='kaolin',
    version='0.0.1',
    url='https://github.com/jldaniel/Kaolin',
    packages=find_packages(),
    author='Jason Daniel',
    author_email="jdanielae@gmail.com",
    description='Guassian process regression with kernel search',
    install_requires=[
        "numpy >= 1.15.0",
        "scipy >= 1.1.0",
        "scikit-learn >= 0.19.2",
    ],
)
