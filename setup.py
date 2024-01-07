from setuptools import setup, find_packages

import os


def read(path):
    return open(os.path.join(os.path.dirname(__file__), path)).read()


setup(
    name='efusor',
    url='https://github.com/esrel/efusor',
    version='0.1.1',
    author='Evgeny A. Stepanov',
    author_email='stepanov.evgeny.a@gmail.com',
    description='Extended Decision Fusion',
    readme="README.md",
    long_description=read('README.md'),
    long_description_content_type="text/markdown",
    package_dir={'': "src"},
    packages=find_packages('src'),
    classifiers=['Development Status :: 3 - Alpha'],
    python_requires='>=3.10'
)
