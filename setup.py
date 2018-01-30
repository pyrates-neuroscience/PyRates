from setuptools import setup, find_packages

# fixme: folder reference needs to be fixed

setup(name='PyRates',
      version='0.2',
      description='Neural Mass Modeling Framework',
      author='Richard Gast, Daniel Rose, Konstantin Weise',
      author_email='rgast@cbs.mpg.de',
      license='GPL3',
      packages=['core/axon',
                'core/synapse',
                'core/population',
                'core/circuit',
                'core/utility',
                'tests'],
      zip_safe=False,
      python_requires='>=3.5',
      install_requires=['numpy', 'matplotlib', 'scipy', 'networkx'])
