from setuptools import setup, find_packages

LIBRARY_VERSION = "0.5.0"

PYRATES_TEAM = "Richard Gast, Daniel Rose"

INSTALL_REQUIREMENTS = ['numpy',
                        'matplotlib',
                        'networkx',
                        'pandas',
                        'tensorflow>=1.12',
                        'pyparsing',
                        'ruamel.yaml',
                        'scipy',
                        'seaborn',
                        'mne',
                        'pydot']

CLASSIFIERS = ["Programming Language :: Python :: 3",
               "Programming Language :: Python :: 3.6",
               "Programming Language :: Python :: 3.7",
               "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
               "Operating System :: OS Independent",
               "Development Status :: 3 - Alpha",
               "Intended Audience :: Developers",
               "Intended Audience :: Science/Research",
               "Topic :: Scientific/Engineering",
               ]

with open("README.md", "r") as fh:
    DESCRIPTION = fh.read()

setup(name='pyrates',
      version=LIBRARY_VERSION,
      description='Neural Network Modeling Framework',
      long_description=DESCRIPTION,
      author=PYRATES_TEAM,
      author_email='rgast@cbs.mpg.de',
      license='GPL v3',
      packages=find_packages(),
      zip_safe=False,
      python_requires='>=3.6',
      install_requires=INSTALL_REQUIREMENTS,
      classifiers=CLASSIFIERS
      )
