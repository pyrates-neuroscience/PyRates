from setuptools import setup, find_packages

# fixme: folder reference needs to be fixed

setup(name='PyRates',
      version='0.3',
      description='Neural Mass Modeling Framework',
      author='Richard Gast, Daniel Rose, Konstantin Weise',
      author_email='rgast@cbs.mpg.de',
      license='GPL3',
      packages=['pyrates/axon',
                'pyrates/synapse',
                'pyrates/population',
                'pyrates/circuit',
                'pyrates/utility',
                'tests',
                'models'],
      zip_safe=False,
      python_requires='>=3.6',
      install_requires=['numpy',
                        'matplotlib',
                        'scipy',
                        'networkx',
                        'pandas',
                        'tensorflow',
                        'ruamel.yaml'])
