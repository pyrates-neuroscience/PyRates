from setuptools import setup, find_packages

PYRATES_TEAM = "Richard Gast, Daniel Rose"

INSTALL_REQUIREMENTS = ['numpy',
                        'networkx',
                        'pandas',
                        'pyparsing',
                        'ruamel.yaml',
                        ]

CLASSIFIERS = ["Programming Language :: Python :: 3",
               "Programming Language :: Python :: 3.6",
               "Programming Language :: Python :: 3.7",
               "Programming Language :: Python :: 3.8",
               "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
               "Operating System :: OS Independent",
               "Development Status :: 3 - Alpha",
               "Intended Audience :: Developers",
               "Intended Audience :: Science/Research",
               "Topic :: Scientific/Engineering",
               ]

EXTRAS = {"tf": ["tensorflow>=2.0"],
          "plot": ["seaborn", "pydot"],
          "proc": ["mne", "scipy"],
          "cluster": ["paramiko", "h5py"],
          "numba": ["numba"],
          "dev": ["pytest", "bump2version"]}

EXTRAS["all"] = [item for sublist in EXTRAS.values() for item in sublist]
EXTRAS["tests"] = [item for key, sublist in EXTRAS.items() for item in sublist if key in ("tf", "proc", "dev")]

with open("VERSION", "r", encoding="utf8") as fh:
    VERSION = fh.read().strip()

with open("README.md", "r", encoding="utf8") as fh:
    DESCRIPTION = fh.read()

setup(name='pyrates',
      version=VERSION,
      description='Neural Network Modeling Framework',
      long_description=DESCRIPTION,
      author=PYRATES_TEAM,
      author_email='rgast@cbs.mpg.de',
      license='GPL v3',
      packages=find_packages(),
      zip_safe=False,
      python_requires='>=3.6',
      install_requires=INSTALL_REQUIREMENTS,
      extras_require=EXTRAS,
      classifiers=CLASSIFIERS,
      include_package_data=True  # include additional non-python files specified in MANIFEST.in
      )
