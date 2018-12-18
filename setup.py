from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='pyrates',
      version='0.5.0',
      description='Neural Network Modeling Framework',
      long_description=long_description,
      author='Richard Gast, Daniel Rose',
      author_email='rgast@cbs.mpg.de',
      license='GPLv3',
      packages=find_packages(),
      zip_safe=False,
      python_requires='>=3.6',
      install_requires=['numpy',
                        'matplotlib',
                        'scipy',
                        'networkx',
                        'pandas',
                        'tensorflow',
                        'pyparsing',
                        'seaborn',
                        'mne',
                        'yaml',
                        'pydot'])
