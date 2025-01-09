from setuptools import setup, find_packages

with open('README.md', encoding='utf-8') as f:
    long_description = f.read()


setup(
    name='MedVLM',
    author="Gustav Müller-Franzes",
    version="1.0",
    description="Code for MedVLM", 
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),
)